import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#verinin dosya yolunu ekleme
import pandas as pd
diabete=pd.read_csv("../input/testdiabete/diabete.csv")
#verinin ilk 5 satırını görüntüleme
diabete.head()
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
print(diabete.shape)
diabete.info()
diabete.describe()
print(diabete["age"].describe)
diabete["notp"].value_counts()
diabete.sort_values(by='age', ascending=False).head(10)