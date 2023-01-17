# Verileri işlemek için Numpy ve Pandas Kütüphanelerini yüklüyoruz
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np

# Bölümler arasında uyarı olmaması için uyarıları kapatıyoruz
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 60)

# Görselleştirme için matplotlib kütüphanesini dahil ediyoruz
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(font_scale = 2)
# Yazı boyutlarını varsayılan  olarak 24 punto ayarlıyoruz
plt.rcParams['font.size'] = 24

from IPython.core.pylabtools import figsize

# Eksik değerlere etki edecek kütüphaneyi yüklüyoruz
from sklearn.preprocessing import Imputer, MinMaxScaler

# Makine Öğrenmesi modellerini dahil ediyoruz
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree

#Tahminleri açıklamak için  LIME kütüphanesini ekliyoruz
import lime
import lime.lime_tabular
# Veriçerçevesinden verileri okuyoruz
train_features = pd.read_csv('../input/nyctrain-test-data/training_features.csv')
test_features = pd.read_csv('../input/nyctrain-test-data/testing_features.csv')
train_labels = pd.read_csv('../input/nyctrain-test-data/training_labels.csv')
test_labels = pd.read_csv('../input/nyctrain-test-data/testing_labels.csv')
#Median doldurma stratejisi ile bir imputer nesne oluşturuyoruz
imputer = Imputer(strategy = 'median')

#Eğitim verilerine imputer nesnesini uyguluyoruz
imputer.fit(train_features)

#Eğtim ve Test verilerini dönüştürüyoruz
X = imputer.transform(train_features)
X_test = imputer.transform(test_features)

#Tek boyutlu vektör için Sklearn de etiketleme yapıyoruz
y = np.array(train_labels).reshape((-1,))
y_test = np.array(test_labels).reshape((-1,))
#Ortalama mutlak hata'yı(MAE) bulmak için fonksiyonumuz
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))
model = GradientBoostingRegressor(loss = 'lad', max_depth = 5, max_features = None,
                                 min_samples_leaf = 6, min_samples_split = 6,
                                 n_estimators = 800, random_state = 42)
model.fit(X, y)
#Test verileri üzerinde tahminlerimizi yapıyoruz
model_pred = model.predict(X_test)

print('Test verileri üzerinde son modelimizin performans testi: MAE = %0.4f' % mae(y_test, model_pred))
#Veri çerçevesi içerisinden önemli özellikleri ayırıyoruz
features_results = pd.DataFrame({'features': list(train_features.columns),
                                'importance': model.feature_importances_})
#içlerinden en önemli 10 tanesine bakalım
features_results = features_results.sort_values('importance', ascending = False).reset_index(drop = True)

features_results.head(10)
figsize(12, 10)
plt.style.use('fivethirtyeight')

#En önemli 20 özelliği yatay bar çubuğundan görselleştirelim
features_results.loc[:9, :].plot(x = 'features', y = 'importance',
                               edgecolor = 'k',
                               kind = 'barh', color = 'yellow');
plt.xlabel('Bağıl Önem', size = 20); 
plt.ylabel('')
plt.title('Rastgele Ormanda Özelliklerin önemi', size = 30);
#En Önemli Özellikleri ayıklıyoruz
most_important_features = features_results['features'][:10]

#Herbir özellik isminin indexini bulalım
indices = [list(train_features.columns).index(x) for x in most_important_features]

#Sadece en önemli özellikler kalıyor
X_reduced =X[:, indices]
X_test_reduced = X_test[:, indices]

print('Eğitim setindeki en önemli özelliklerin şekli: ', X_reduced.shape)
print('Test setindeki en önemli özelliklerin şekli: ', X_test_reduced.shape)
lr = LinearRegression()

#Özelliklerin tamamına uyguluyoruz
lr.fit(X, y)
lr_full_pred = lr.predict(X_test)

#Azaltılmış özelliklere uyguluyoruz
lr.fit(X_reduced, y)
lr_reduced_pred = lr.predict(X_test_reduced)

#Sonuçları görüntülüyoruz
print('Doğrusal Regresyonun tüm sonuçları: MAE =  %0.4f.' % mae(y_test, lr_full_pred))
print('Doğrusal Regresyonun azaltılmış sonuçları : MAE = %0.4f.' % mae(y_test, lr_reduced_pred))
#Aynı hiperparametreler'den oluşan bir model oluşturuyoruz
model_reduced = GradientBoostingRegressor(loss = 'lad', max_depth = 5, max_features = None,
                                       min_samples_leaf = 6, min_samples_split =6, 
                                       n_estimators = 800, random_state = 42)
#Özellikleri uygulayıp test ediyoruz
model_reduced.fit(X_reduced, y)
model_reduced_pred = model_reduced.predict(X_test_reduced)

print('Eğimi artırılmışların azaltılmış sonuçları: MAE = %0.4f' % mae(y_test, model_reduced_pred))
#Azaltılmış değerleri buluyoruz
residuals = abs(model_reduced_pred - y_test)

#En iyi ve en kötü sonuçları görüntülüyoruz
wrong = X_test_reduced[np.argmax(residuals), :]
right = X_test_reduced[np.argmin(residuals), :]
# Açıklayıcı(explainer) nesnesi oluşturuyoruz
explainer = lime.lime_tabular.LimeTabularExplainer(training_data = X_reduced, 
                                                   mode = 'regression',
                                                   training_labels = y,
                                                   feature_names = list(most_important_features))
# Display the predicted and true value for the wrong instance
print('Prediction: %0.4f' % model_reduced.predict(wrong.reshape(1, -1)))
print('Actual Value: %0.4f' % y_test[np.argmax(residuals)])

# Explanation for wrong prediction
wrong_exp = explainer.explain_instance(data_row = wrong, 
                                       predict_fn = model_reduced.predict)

# Plot the prediction explaination
wrong_exp.as_pyplot_figure();
plt.title('Explanation of Prediction', size = 28);
plt.xlabel('Effect on Prediction', size = 22);
wrong_exp.show_in_notebook(show_predicted_value=False)
# Tahmin edilen ve gerçek değerlerin nasıl göründüğüne bakalım
print('Tahmin: %0.4f' % model_reduced.predict(right.reshape(1, -1)))
print('Gerçek Değer: %0.4f' % y_test[np.argmin(residuals)])

# Yanlış tahminin açıklaması
right_exp = explainer.explain_instance(right, model_reduced.predict, num_features=10)
right_exp.as_pyplot_figure();
plt.title('Tahminin Açıklaması', size = 28);
plt.xlabel('Tahmin üzerindeki etkisi', size = 22);
right_exp.show_in_notebook(show_predicted_value=False)