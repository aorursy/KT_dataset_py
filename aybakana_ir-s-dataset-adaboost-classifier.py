from sklearn.ensemble import AdaBoostClassifier

from sklearn import datasets



from sklearn.model_selection import train_test_split



from sklearn import metrics

from sklearn.svm import SVC

svc=SVC(probability=True, kernel='linear')

iris = datasets.load_iris()

X = iris.data

y = iris.target

X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.3)



# adaboost sınıflandırıcı objesini oluşturma

#abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)

abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)



# Adaboost Sınıflandırıcı Eğitimi

model = abc.fit(X_egitim, y_egitim)



# Test veri seti için modeli çalıştıralım

y_tahmin = model.predict(X_test)



# Modelin Doğruluk oranı

print("Dogruluk:",metrics.accuracy_score(y_test, y_tahmin))