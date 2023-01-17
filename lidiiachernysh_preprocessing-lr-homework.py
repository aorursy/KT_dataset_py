import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')
%matplotlib inline
%pylab inline
data = pd.read_csv('../input/data.csv')
data.shape
X = data.drop('Grant.Status', 1)
y = data['Grant.Status']
data.head(10)
numeric_cols = ['RFCD.Percentage.1', 'RFCD.Percentage.2', 'RFCD.Percentage.3', 
                'RFCD.Percentage.4', 'RFCD.Percentage.5',
                'SEO.Percentage.1', 'SEO.Percentage.2', 'SEO.Percentage.3',
                'SEO.Percentage.4', 'SEO.Percentage.5',
                'Year.of.Birth.1', 'Number.of.Successful.Grant.1', 'Number.of.Unsuccessful.Grant.1']
categorical_cols = list(set(X.columns.values.tolist()) - set(numeric_cols))
data.dropna().shape
def calculate_means(numeric_data):
    means = np.zeros(numeric_data.shape[1])
    for j in range(numeric_data.shape[1]):
        to_sum = numeric_data.iloc[:,j]
        indices = np.nonzero(~numeric_data.iloc[:,j].isnull())[0]
        correction = np.amax(to_sum[indices])
        to_sum /= correction
        for i in indices:
            means[j] += to_sum[i]
        means[j] /= indices.size
        means[j] *= correction
    return pd.Series(means, numeric_data.columns)
# place your code here

X_real_zeros = X[numeric_cols].fillna(0.0)
X_real_mean = X[numeric_cols].fillna(calculate_means(X[numeric_cols]))
X_cat = X[categorical_cols].fillna('NA').astype('str')

#X_cat = X[categorical_cols].fillna('NA').applymap(str)
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction import DictVectorizer as DV

categorial_data = pd.DataFrame({'sex': ['male', 'female', 'male', 'female'], 
                                'nationality': ['American', 'European', 'Asian', 'European']})
print('Исходные данные:\n')
print(categorial_data)
encoder = DV(sparse = False)
encoded_data = encoder.fit_transform(categorial_data.T.to_dict().values())
print('\nЗакодированные данные:\n')
print(encoded_data)
encoder = DV(sparse = False)
X_cat_oh = encoder.fit_transform(X_cat.T.to_dict().values())
from sklearn.model_selection import train_test_split

(X_train_real_zeros, 
 X_test_real_zeros, 
 y_train, y_test) = train_test_split(X_real_zeros, y, 
                                     test_size=0.3, 
                                     random_state=0)
(X_train_real_mean, 
 X_test_real_mean) = train_test_split(X_real_mean, 
                                      test_size=0.3, 
                                      random_state=0)
(X_train_cat_oh,
 X_test_cat_oh) = train_test_split(X_cat_oh, 
                                   test_size=0.3, 
                                   random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

def plot_scores(optimizer):
    scores = [] 
    for i in range(len(optimizer.cv_results_['params'])):
        scores.append([optimizer.cv_results_['params'][i]['C'], 
                       optimizer.cv_results_['mean_test_score'][i], 
                       optimizer.cv_results_['std_test_score'][i]])
        
    scores = np.array(scores)
    plt.semilogx(scores[:, 0], scores[:, 1])
    plt.fill_between(scores[:, 0], scores[:, 1] - scores[:, 2],
                                  scores[:, 1] + scores[:, 2], alpha=0.3)
    plt.show()
 
    
def write_answer_1(auc_1, auc_2):
    auc = (auc_1 + auc_2)/2
    with open("preprocessing_lr_answer1.txt", "w") as fout:
        fout.write(str(auc))
        
param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
cv = 3

# place your code here

X_train_zeros = np.hstack((X_train_real_zeros, X_train_cat_oh))
X_train_mean = np.hstack((X_train_real_mean, X_train_cat_oh))

optimizer_zero = GridSearchCV(LogisticRegression(), param_grid, cv=cv, n_jobs=-1)
optimizer_mean = GridSearchCV(LogisticRegression(), param_grid, cv=cv, n_jobs=-1)
%%time 
optimizer_mean.fit(X_train_mean, y_train)
%%time 
optimizer_zero.fit(X_train_zeros, y_train)
print(optimizer_zero.best_score_)
print(optimizer_zero.best_params_)
print(optimizer_zero.best_estimator_)

#plot_scores(optimizer_zero)
print(optimizer_mean.best_score_)
print(optimizer_mean.best_params_)
print(optimizer_mean.best_estimator_)

#plot_scores(optimizer_mean)
X_test_zeros = np.hstack((X_test_real_zeros, X_test_cat_oh))
X_test_mean = np.hstack((X_test_real_mean, X_test_cat_oh))

predict_zero = optimizer_zero.predict_proba(X_test_zeros)
predict_mean = optimizer_mean.predict_proba(X_test_mean)
roc_auc_zero = roc_auc_score(y_test.T, predict_zero[:, 1])
roc_auc_mean = roc_auc_score(y_test, predict_mean[:, 1])

print(roc_auc_zero)
print(roc_auc_mean)
#write_answer_1(roc_auc_zero, roc_auc_mean)
from pandas.plotting import scatter_matrix

data_numeric = pd.DataFrame(X_train_real_mean, columns=numeric_cols)
list_cols = ['Number.of.Successful.Grant.1', 'SEO.Percentage.2', 'Year.of.Birth.1']
scatter_matrix(data_numeric[list_cols], alpha=0.5, figsize=(10, 10))
plt.show()
from sklearn.preprocessing import StandardScaler

# place your code here
scaler = StandardScaler()
X_train_real_scaled = StandardScaler.fit_transform(scaler, X_train_real_mean)
X_test_real_scaled = StandardScaler.fit_transform(scaler, X_test_real_mean)
data_numeric_scaled = pd.DataFrame(X_train_real_scaled, columns=numeric_cols)
list_cols = ['Number.of.Successful.Grant.1', 'SEO.Percentage.2', 'Year.of.Birth.1']
scatter_matrix(data_numeric_scaled[list_cols], alpha=0.5, figsize=(10, 10))
plt.show()
def write_answer_2(auc):
    with open("preprocessing_lr_answer2.txt", "w") as fout:
        fout.write(str(auc))
        
# place your code here

X_train_scaled = np.hstack((X_train_real_scaled, X_train_cat_oh))
X_test_scaled = np.hstack((X_test_real_scaled, X_test_cat_oh))
%%time 
optimizer_scale= GridSearchCV(LogisticRegression(), param_grid, cv=cv, n_jobs=-1)
optimizer_scale.fit(X_train_scaled, y_train)
print(optimizer_scale.best_score_)
print(optimizer_scale.best_params_)
print(optimizer_scale.best_estimator_)
plot_scores(optimizer_scale)
X_test_scale = np.hstack((X_test_real_scaled, X_test_cat_oh))

predict_scale_test = optimizer_scale.predict_proba(X_test_scale)
roc_auc_scale = roc_auc_score(y_test.T, predict_scale_test[:, 1])

print(roc_auc_scale)

#write_answer_2(roc_auc_scale)
np.random.seed(0)
"""Сэмплируем данные из первой гауссианы"""
data_0 = np.random.multivariate_normal([0,0], [[0.5,0],[0,0.5]], size=40)
"""И из второй"""
data_1 = np.random.multivariate_normal([0,1], [[0.5,0],[0,0.5]], size=40)

"""На обучение берём 20 объектов из первого класса и 10 из второго"""
example_data_train = np.vstack([data_0[:20,:], data_1[:10,:]])
example_labels_train = np.concatenate([np.zeros((20)), np.ones((10))])

"""На тест - 20 из первого и 30 из второго"""
example_data_test = np.vstack([data_0[20:,:], data_1[10:,:]])
example_labels_test = np.concatenate([np.zeros((20)), np.ones((30))])

"""Задаём координатную сетку, на которой будем вычислять область классификации"""
xx, yy = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))

"""Обучаем регрессию без балансировки по классам"""
optimizer = GridSearchCV(LogisticRegression(), param_grid, cv=cv, n_jobs=-1)
optimizer.fit(example_data_train, example_labels_train)

"""Строим предсказания регрессии для сетки"""
Z = optimizer.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
plt.scatter(data_0[:,0], data_0[:,1], color='red')
plt.scatter(data_1[:,0], data_1[:,1], color='blue')

"""Считаем AUC"""
auc_wo_class_weights = roc_auc_score(example_labels_test, optimizer.predict_proba(example_data_test)[:,1])
plt.title('Without class weights')
plt.show()
print('AUC: %f'%auc_wo_class_weights)

"""Для второй регрессии в LogisticRegression передаём параметр class_weight='balanced'"""
optimizer = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=cv, n_jobs=-1)
optimizer.fit(example_data_train, example_labels_train)

Z = optimizer.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
plt.scatter(data_0[:,0], data_0[:,1], color='red')
plt.scatter(data_1[:,0], data_1[:,1], color='blue')

auc_w_class_weights = roc_auc_score(example_labels_test, optimizer.predict_proba(example_data_test)[:,1])
plt.title('With class weights')
plt.show()
print('AUC: %f'%auc_w_class_weights)
print(np.sum(y_train==0))
print(np.sum(y_train==1))
def write_answer_3(auc_1, auc_2):
    auc = (auc_1 + auc_2) / 2
    with open("preprocessing_lr_answer3.txt", "w") as fout:
        fout.write(str(auc))
        
# place your code here
optimizer_balanced = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=cv, n_jobs=-1)
optimizer_balanced.fit(X_train_scaled, y_train)

predict_balanced = optimizer_balanced.predict_proba(X_test_scaled)
auc_class_weight = roc_auc_score(y_test.T, predict_balanced[:, 1])
print(auc_class_weight)
np.random.seed(0)
indices_to_add = np.random.randint(np.sum(y_train==1), size=np.sum(y_train==0) - np.sum(y_train==1))
X_train_to_add = X_train_scaled[y_train.as_matrix()==1,:][indices_to_add,:]
y_train_to_add = np.repeat(1, np.sum(y_train==0) - np.sum(y_train==1)).T

X_train_added = np.vstack((X_train_scaled, X_train_to_add))
y_train_added = np.hstack([y_train, y_train_to_add])
print(shape(X_train_to_add))

print(shape(X_train_added))
print(shape(y_train_added))
optimizer_ad = GridSearchCV(LogisticRegression(), param_grid, cv=cv, n_jobs=-1)
optimizer_ad.fit(X_train_added, y_train_added)

predict_ad = optimizer_ad.predict_proba(X_test_scaled)
auc_class_ad = roc_auc_score(y_test.T, predict_ad[:, 1])
print(auc_class_ad)

#write_answer_3(auc_class_weight, auc_class_ad)
print('AUC ROC for classifier without weighted classes', auc_wo_class_weights)
print('AUC ROC for classifier with weighted classes: ', auc_w_class_weights)
"""Разделим данные по классам поровну между обучающей и тестовой выборками"""
example_data_train = np.vstack([data_0[:20,:], data_1[:20,:]])
example_labels_train = np.concatenate([np.zeros((20)), np.ones((20))])
example_data_test = np.vstack([data_0[20:,:], data_1[20:,:]])
example_labels_test = np.concatenate([np.zeros((20)), np.ones((20))])
"""Обучим классификатор"""
optimizer = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=cv, n_jobs=-1)
optimizer.fit(example_data_train, example_labels_train)
Z = optimizer.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
plt.scatter(data_0[:,0], data_0[:,1], color='red')
plt.scatter(data_1[:,0], data_1[:,1], color='blue')
auc_stratified = roc_auc_score(example_labels_test, optimizer.predict_proba(example_data_test)[:,1])
plt.title('With class weights')
plt.show()
print('AUC ROC for stratified samples: ', auc_stratified)
def write_answer_4(auc):
    with open("preprocessing_lr_answer4.txt", "w") as fout:
        fout.write(str(auc))
        
# place your code here
from sklearn.model_selection import train_test_split

(x_train_real, x_test_real, 
 y_train_real, y_test_real) = train_test_split(X_real_zeros, y, 
                                               test_size=0.3, 
                                               stratify=y, 
                                               random_state=0)

(x_train_cat, x_test_cat) = train_test_split(X_cat_oh, 
                                             test_size=0.3,
                                             stratify=y,
                                             random_state=0)
x_train_real_scaled = StandardScaler.fit_transform(scaler, x_train_real)
x_test_real_scaled = StandardScaler.fit_transform(scaler, x_test_real)
x_train_scaled = np.hstack((x_train_real_scaled, x_train_cat))
x_test_scaled = np.hstack((x_test_real_scaled, x_test_cat))
optimizer_b = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=cv, n_jobs=-1)
optimizer_b.fit(x_train_scaled, y_train_real)
predict_b = optimizer_b.predict_proba(x_test_scaled)
auc_weight = roc_auc_score(y_test_real, predict_b[:, 1])
print(auc_weight)

write_answer_4(auc_weight)
from sklearn.preprocessing import PolynomialFeatures

"""Инициализируем класс, который выполняет преобразование"""
transform = PolynomialFeatures(2)
"""Обучаем преобразование на обучающей выборке, применяем его к тестовой"""
example_data_train_poly = transform.fit_transform(example_data_train)
example_data_test_poly = transform.transform(example_data_test)
"""Обращаем внимание на параметр fit_intercept=False"""
optimizer = GridSearchCV(LogisticRegression(class_weight='balanced', fit_intercept=False), param_grid, cv=cv, n_jobs=-1)
optimizer.fit(example_data_train_poly, example_labels_train)
Z = optimizer.predict(transform.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
plt.scatter(data_0[:,0], data_0[:,1], color='red')
plt.scatter(data_1[:,0], data_1[:,1], color='blue')
plt.title('With class weights')
plt.show()
print(example_data_train_poly.shape)
transform = PolynomialFeatures(11)
example_data_train_poly = transform.fit_transform(example_data_train)
example_data_test_poly = transform.transform(example_data_test)
optimizer = GridSearchCV(LogisticRegression(class_weight='balanced', fit_intercept=False), param_grid, cv=cv, n_jobs=-1)
optimizer.fit(example_data_train_poly, example_labels_train)
Z = optimizer.predict(transform.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel2)
plt.scatter(data_0[:,0], data_0[:,1], color='red')
plt.scatter(data_1[:,0], data_1[:,1], color='blue')
plt.title('Corrected class weights')
plt.show()
print(example_data_train_poly.shape)
def write_answer_5(auc):
    with open("preprocessing_lr_answer5.txt", "w") as fout:
        fout.write(str(auc))
        
# place your code here
from sklearn.preprocessing import PolynomialFeatures

transform = PolynomialFeatures(2)
x_train_real_poly = transform.fit_transform(x_train_real)
x_test_real_poly = transform.fit_transform(x_test_real)

print(x_train_real_poly)

x_train_poly_scaled = StandardScaler.fit_transform(scaler, x_train_real_poly)
x_test_poly_scaled = StandardScaler.fit_transform(scaler, x_test_real_poly)
x_train_stack = np.hstack((x_train_poly_scaled, x_train_cat))
x_test_stack = np.hstack((x_test_poly_scaled, x_test_cat))

print(shape(x_train_stack))
print(shape(x_test_stack))
optimizer_poly = GridSearchCV(LogisticRegression(class_weight='balanced', fit_intercept=False), param_grid, cv=cv, n_jobs=-1)
optimizer_poly.fit(x_train_stack, y_train_real)
x_test_stack
pred = optimizer_poly.predict_proba(x_test_stack)
print(pred)
auc_weight = roc_auc_score(y_test_real, pred[:, 1])
print(auc_weight)

write_answer_5(auc_weight)
def write_answer_6(features):
    with open("preprocessing_lr_answer6.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in features]))
        
# place your code here
optimizer_lasso = GridSearchCV(LogisticRegression(class_weight='balanced',
                                                  penalty='l1', 
                                                  solver='liblinear'), 
                               param_grid, cv=cv, n_jobs=-1)

optimizer_lasso.fit(x_train_scaled, y_train_real)

predict_lasso = optimizer_lasso.predict_proba(x_test_scaled)
auc_weight = roc_auc_score(y_test_real, predict_lasso[:, 1])
print(auc_weight)
coef = optimizer_lasso.best_estimator_.coef_[0][:13]
coef
feat = np.where(coef == 0)
feat[0]
write_answer_6(feat[0])
