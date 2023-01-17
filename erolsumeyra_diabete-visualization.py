import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#verinin dosya yolunu ekleme
import pandas as pd
diabete=pd.read_csv("../input/testdiabete/diabete.csv")
import csv

header = ["notp", "pga", "dip","tsft","2hoursi","bmi","def","age","Class"]

with open('../input/testdiabete/diabete.csv', 'r') as fp:
    reader = csv.DictReader(fp, fieldnames=header)


    with open('diabete.csv', 'w', newline='') as fh: 
        writer = csv.DictWriter(fh, fieldnames=reader.fieldnames)
        writer.writeheader()
        header_mapping = next(reader)
        writer.writerows(reader)
diabete=pd.read_csv("diabete.csv")
diabete.head()
diabete.info()
diabete["Class"].hist()
plt.hist(diabete.bmi,bins=12,color='#acac00',
         histtype="stepfilled",label="Vücut Kitle İndexi")

plt.xlabel("Oranlar")
plt.ylabel("Dağılımlar")
plt.legend()
plt.title("Vücut Kitle İndexi Dağılımı")
plt.show()
plt.hist([diabete.bmi,diabete.age],
         color=['#0c457d','#0ea7b5'],
         label=["bmi","age"])

plt.xlabel("Oranlar")

plt.ylabel("Dağılımlar")
plt.title("Vücut Kitle İndexi ve Yaş Oranları")
plt.legend() 
plt.show()
%matplotlib inline
import matplotlib.pyplot as plt
diabete.hist(bins=50, figsize=(20,15))
plt.show()
DiyabetOlmayan = len(diabete[diabete.Class == 0])
DiyabetOlan = len(diabete[diabete.Class == 1])
print("Diyabet Olmayanların Yüzdesi: {:.2f}%".format((DiyabetOlmayan / (len(diabete.Class))*100)))
print("Diyabet Olanların Yüzdesi: {:.2f}%".format((DiyabetOlan / (len(diabete.Class))*100)))
pd.crosstab(diabete.age,diabete.Class).plot(kind="bar",figsize=(20,10))
plt.title('Yaşa göre Diyabet Hastaları Dağılımı')
plt.xlabel('Yaş')
plt.ylabel('Sıklık')
plt.show()
diabete.groupby('age')['Class'].describe()
plt.figure(figsize=(12,4))
pd.crosstab(diabete.notp,diabete.age).plot(kind="bar",figsize=(20,10))
plt.title('Hamile Sayısına Göre Yaş Dağılımı')
plt.xlabel('Hamile Sayısı')
plt.ylabel('Sıklık')
plt.savefig(fname="hamile.png",facecolor="green")
plt.show()