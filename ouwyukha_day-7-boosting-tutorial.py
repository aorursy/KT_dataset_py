import pandas as pd # Data manipulation and analysis tools

from matplotlib import pyplot as plt # Class untuk menampilkan grafik



from sklearn.model_selection import train_test_split # Fungsi untuk memecah dataset
from sklearn import metrics



def metric_evaluation(y_test, y_pred):

    print("Accuracy :", metrics.accuracy_score(y_test, y_pred))

    print("Precision:", metrics.precision_score(y_test, y_pred))

    print("Recall   :", metrics.recall_score(y_test, y_pred))

    print("F1-score :", metrics.f1_score(y_test, y_pred))
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/us-adult-census-dataset-cleaned/adult_train.csv') # Membaca data `adult_train.csv`

test = pd.read_csv('/kaggle/input/us-adult-census-dataset-cleaned/adult_test.csv') # Membaca data `adult_test.csv`

train.shape, test.shape # Menampilkan ukuran dataset
train.head(5) # Menampilkan 5 data teratas dari dataset
train.info() # Menampilkan informasi dasar dari dataset
# Membuang kelima fitur di atas dari trainset dan testset

train.drop(columns=['fnlwgt', 'relationship', 'capital_gain', 'capital_loss', 'education_num'], inplace=True)

test.drop(columns=['fnlwgt', 'relationship', 'capital_gain', 'capital_loss', 'education_num'], inplace=True)
# Pertama, mari kita pisahkan data fitur dengan target label-nya.

X_train = train.drop(columns=['label'])

y_train = train['label'].values.reshape(-1)



X_test = test.drop(columns=['label'])

y_test = test['label'].values.reshape(-1)
# Selanjutnya, kita akan memecah trainset menjadi trainset dan validset

# dengan proporsi validset sebanyak 20% dari keseluruhan data.

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
# Untuk menampilkan jumlah data tiap set bisa menggunakan fungsi len()

len(X_train), len(X_valid), len(X_test)
# One Hot Encode ketiga dataset menggunakan pd.get_dummies()

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)



# Lalu mengkalibrasi X_test dan X_valid dengan nama-nama fitur dari X_train

X_train, X_test = X_train.align(X_test, join='left', axis=1)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)



# Mengisi fitur baru hasil kalibrasi dengan 0

X_test.fillna(0, inplace=True)

X_valid.fillna(0, inplace=True)



display(X_train.head(3))
# Import class XGBClassifier dari library xgboost

from xgboost import XGBClassifier
# Inisialisasi model XGBoost dengan parameter random_state untuk menghasilkan perilaku yang dapat di reproduce.

xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X=X_train, y=y_train)
# Prediksi testset dengan fungsi .predict()

xgb_pred = xgb_model.predict(X_test)
# Memanggil fungsi metric_evaluation yang telah kita buat untuk menghitung evaluation score

print('XGBoost')

metric_evaluation(y_test, xgb_pred)
xgb_model = XGBClassifier(

    booster='gbtree',

    n_estimators=100,

    max_depth=6,

    learning_rate=0.3,

    objective='binary:logistic',

    n_jobs=-1,

    random_state=0

)
xgb_model.fit(X=X_train,

              y=y_train,

              eval_set=[(X_train, y_train), (X_valid, y_valid)], # dataset yang ingin dievaluasi

              eval_metric=['logloss'], # Metode pengukuran evaluasi

              early_stopping_rounds=10, # Jumlah iterasi sebelum model memutuskan untuk berhenti

              verbose=True

             )
xgb_evals_result = xgb_model.evals_result_ # Mengambil hasil evaluasi saat training



plt.plot(xgb_evals_result['validation_0']['logloss'], label='train') # Membuat garis logloss trainset

plt.plot(xgb_evals_result['validation_1']['logloss'], label='valid') # Membuat garis logloss validset

plt.xlabel('Iteration') # Memberi label pada koordinat x

plt.ylabel('logloss') # Memberi label pada koordinat y

plt.title('Training Trajectory') # Memberi judul pada grafik

plt.legend() # Menampilkan legenda seperti label-label tadi

plt.show() # Menampilkan plot
# Prediksi testset dengan fungsi .predict()

xgb_pred = xgb_model.predict(X_test)
# Memanggil fungsi metric_evaluation yang telah kita buat untuk menghitung evaluation score

print('XGBoost')

metric_evaluation(y_test, xgb_pred)
# Import class LGBMClassifier dari library lightgbm

from lightgbm import LGBMClassifier
# Inisialisasi model LightGBM dengan parameter random_state untuk menghasilkan perilaku yang dapat di reproduce.

lgb_model = LGBMClassifier(random_state=0)
lgb_model.fit(X=X_train, y=y_train)
# Prediksi testset dengan fungsi .predict()

lgb_pred = lgb_model.predict(X_test)
# Memanggil fungsi metric_evaluation yang telah kita buat untuk menghitung evaluation score

print('LightGBM')

metric_evaluation(y_test, lgb_pred)
lgb_model = LGBMClassifier(

    boosting_type='gbdt',

    n_estimators=100,

    max_depth=-1,

    learning_rate=0.1,

    objective='binary',

    n_jobs=-1,

    random_state=0

)
lgb_model.fit(X=X_train,

              y=y_train,

              eval_set=[(X_train, y_train), (X_valid, y_valid)],

              eval_metric=['binary_logloss'],

              early_stopping_rounds=10,

              verbose=True

             )
lgb_evals_result = lgb_model.evals_result_ # Mengambil hasil evaluasi saat training



plt.plot(lgb_evals_result['training']['binary_logloss'], label='train') # Membuat garis logloss trainset

plt.plot(lgb_evals_result['valid_1']['binary_logloss'], label='valid') # Membuat garis logloss validset

plt.xlabel('Iteration') # Memberi label pada koordinat x

plt.ylabel('binary_logloss') # Memberi label pada koordinat y

plt.title('Training Trajectory') # Memberi judul pada grafik

plt.legend() # Menampilkan legenda seperti label-label tadi

plt.show() # Menampilkan plot
# Prediksi testset dengan fungsi .predict()

lgb_pred = lgb_model.predict(X_test)
# Memanggil fungsi metric_evaluation yang telah kita buat untuk menghitung evaluation score

print('LightGBM')

metric_evaluation(y_test, lgb_pred)