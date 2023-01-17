# Gerekli kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
loans = pd.read_csv("../input/lending-club-data/loan_data.csv")
loans.info()
loans.describe()
loans.head()
# Özet
print("Toplam sütun sayısı (veri öznitelik sayısı):", len(loans.columns))
print("Toplam satır sayısı (veri sayısı):", len(loans.index))
print("Toplam sınıf sayısı: 2 (Borç alanlarının içinden borcunun ödeyenler ve ödemeyenler (1, 0))")
# Kredi politikasının kabul edenler ile etmeyenlerin oldukça faydalı bir veri alan FICO skoruna göre grafiği:
sns.set_style("whitegrid")
fig = plt.figure(figsize=(15,4))
sns.distplot(loans[loans["credit.policy"] == 1]["fico"], kde=False, label="Credit Policy = 1")
sns.distplot(loans[loans["credit.policy"] == 0]["fico"], kde=False, label="Credit Policy = 0")
plt.legend(loc=0)
# Görüldüğü üzere kredi politikasını kabul edenlerin FICO skoru daha yüksek.
# Aynı şekilde çok yararlı FICO bilgisine kullanarak borcunun tamamen ödemeyenlerin FICO değerlerin grafiği:
fig = plt.figure(figsize=(15,4))
sns.distplot(loans[loans["not.fully.paid"] == 0]["fico"], kde=False, label="Not Fully Paid = 0")
sns.distplot(loans[loans["not.fully.paid"] == 1]["fico"], kde=False, label="Not Fully Paid = 1")
plt.legend(loc=0)
# Görüldüğü üzere borcunun tamamen ödemeyenleri FICO değeri daha düşük.
# Borç amaçlarına göre krediyi tamamen ödeme grafiği:
# Hangi amaçla borç alanlar borcunu ödüyor?
fig = plt.figure(figsize=(10,6))
sns.countplot(x="purpose", data=loans, hue="not.fully.paid")
# Bu bir doğal dil işleme problemi olmadığı için metin değerilerini kategori edilebilecek sayısal değerlere dönüştürme:
final_data = pd.get_dummies(loans, columns=["purpose"], drop_first=True)
final_data.head()
# Tamamen sayılardan oluşan bir veri seti elde ettik.
X = final_data.drop("not.fully.paid", axis=1)
y = final_data["not.fully.paid"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
# Veri setini eğitme işlemi:
dtree.fit(X_train, y_train)
# Tahminleme işlemi:
predictions = dtree.predict(X_test)
# Tahminleri kontrol etme işlemi:
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
# Diğer bir test etme işlemi (10 fold cross validation):
from sklearn.model_selection import cross_val_score
print(cross_val_score(dtree, X, y, cv=10))
# 100. sıradaki kişinin özellikleri
final_data.drop("not.fully.paid", axis=1).iloc[100]
# 100. sıradaki kişinin borcunu ödeyip ödemeyeceğini Decision Tree ile tahminleme işlemi:
dtree.predict([final_data.drop("not.fully.paid", axis=1).iloc[100]])
# Tahminlemeye göre kişi borcunu ödemeyecek.
# Gerçek sonuca göre 100. kişi borcunu ödemiyor.
final_data["not.fully.paid"].iloc[100]
# Decision Tree ile yapılan sonuçlar başarılı, şimdi Random Forest ile tahminleme yapalım.
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
# Veri setini eğitme işlemi:
rfc.fit(X_train, y_train)
# Tahminleme işlemi:
rfc_predictions = rfc.predict(X_test)
# Tahminleri kontrol etme işlemi:
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
# Diğer bir test etme işlemi (10 fold cross validation):
print(cross_val_score(rfc, X, y, cv=10))
# 100. sıradaki kişinin özellikleri
final_data.drop("not.fully.paid", axis=1).iloc[100]
# 100. sıradaki kişinin borcunu ödeyip ödemeyeceğini Random Forest ile tahminleme işlemi:
rfc.predict([final_data.drop("not.fully.paid", axis=1).iloc[100]])
# Tahminlemeye göre kişi borcunu ödemeyecek.
# Gerçek sonuca göre 100. kişi borcunu ödemiyor.
final_data["not.fully.paid"].iloc[100]
# Rain Forest ile yapılan sonuçlar da başarılı.