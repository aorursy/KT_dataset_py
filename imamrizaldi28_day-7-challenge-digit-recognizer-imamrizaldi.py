import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/digit-recognizer/train.csv')
X_test = pd.read_csv('../input/digit-recognizer/test.csv')
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
train.shape, X_test.shape, submission.shape
train.head(5)
idx = 41999 # Anda dapat merubah index dari 0 hingga 41999 untuk melihat gambar lainnya
images = train.iloc[:, 1:].values.reshape(42000, 28, 28)
plt.imshow(images[idx])
print('Target Label:', train.iloc[idx, 0])
# Memisahkan Features Data dengan Target Label
X = train.drop(columns=['label'])
y = train[['label']].values.reshape(-1)
# Memecah Trainset menjadi 80% Trainset dan 20% Validset
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
# Mengecek ukuran tiap dataset
len(X_train), len(X_valid), len(X_test)
from lightgbm import LGBMClassifier
parameters = {
    'boosting_type':'gbdt',
    'n_estimators':200,
    'max_depth':3,
    'learning_rate':0.3,
    'n_jobs':-1,
    'random_state':0
}

early_stopping_rounds = 10
model = LGBMClassifier(**parameters)
model.fit(
    X=X_train,
    y=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)], # dataset yang ingin dievaluasi
    eval_metric=['multi_logloss'], # Metode pengukuran evaluasi
    early_stopping_rounds=early_stopping_rounds, # Jumlah iterasi sebelum model memutuskan untuk berhenti
    verbose=True
)
evals_result = model.evals_result_ # Mengambil hasil evaluasi saat training

plt.plot(evals_result['training']['multi_logloss'], label='train') # Membuat garis logloss trainset
plt.plot(evals_result['valid_1']['multi_logloss'], label='valid') # Membuat garis logloss validset
plt.xlabel('Iteration') # Memberi label pada koordinat x
plt.ylabel('multi_logloss') # Memberi label pada koordinat y
plt.title('Training Trajectory') # Memberi judul pada grafik
plt.legend() # Menampilkan legenda seperti label-label tadi
plt.show() # Menampilkan plot
# Fungsi untuk menampilkan 16 gambar acak
def show_prediction(X, y_pred):
    show = np.arange(0, len(y_pred))
    np.random.shuffle(show)
    show = show[:16]

    images = X.iloc[show]
    images = images.values.reshape(-1, 28, 28)
    
    fig=plt.figure(figsize=(7, 7))
    columns = 4
    rows = 4
    for i in range(0, columns*rows):
        ax = fig.add_subplot(rows, columns, i+1)
        ax.set_title('Pred: ' + str(y_pred[show[i]]))
        plt.axis('off')
        plt.imshow(images[i])
    plt.show()
# Memprediksi dapat dilakukan dengan fungsi .predict()
valid_pred = model.predict(X_valid)
# Menampilkan akurasi model dengan fungsi accuracy_score dari sklearn.metrics
accuracy_score(y_valid, valid_pred)
show_prediction(X_valid, valid_pred)
# Prediksi testset dengan fungsi .predict()
y_pred = model.predict(X_test)
show_prediction(X_test, y_pred)
submission['Label'] = y_pred
# Jika sudah, harusnya nilai dari kolom `Label` akan bervariasi, bukan 0 semua
submission.head(5)
submission.to_csv('submission1.csv', index=False) # index=False adalah argumen untuk tidak memasukan index kedalam CSV.
