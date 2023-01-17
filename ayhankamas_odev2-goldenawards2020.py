# Ayhan Kamas tarafından Data Science ve Python dersinin 2. ödevi olarak yazıldı.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
awards=pd.read_csv("/kaggle/input/golden-globe-awards/golden_globe_awards.csv")

awards.info()
awards.tail(10)
# awards["category"].unique()
# son 20 senelik veri seti üzerinde çalışılacağım
awards2000=awards[awards["year_award"]>1999]
#awards2000.head(10)
awards2000.tail(20)
aw=awards.head(3)
aw
# veri setindei satır sayısı
print('Number of rows in the dataset :', awards.shape[0])
# veri setindeki sütun sayısı
print('Number of columns in the dataset :', awards.shape[1])
awards.info()
awards.columns
null_values=awards.isnull().sum() 
null_values=pd.DataFrame(null_values, columns=['Missing Values']) # evet notlarda buraya dikkat pd.DataFrame kullanılacak
null_values  # bu arada run edilince anlaşılacak ki her sütun karşılığında yazan sayı kadar boş veri girilmemiş hücre var
awards["category"].unique()
listNumbersCtg=list(range(awards['category'].nunique())) # nunique()  sanırım unique den kaçtane demek
listCategories=list(awards['category'].unique())
print('---------------  CATEGORIES  -----------------')
for item in zip(listNumbersCtg, listCategories):
    print(item[0], ':', item[1])
    
# Evet listNumberCtg yi sadece 1,2,3... diye sayılarını kullanmak için oluşturmuş. Item[0] onları temsil ediyor

# içinde best taşıyan ödüller - Bunu best film i bulmak için yapıyorum

# BEST FİLMLER BÜTÜN DALLARDA
bestList=[s for s in listCategories if "Best Motion" in s]  
# aranan kelimeyi yazarkem büyük, küçük harf duyarlı old. unutma "best" değil "Best"
for item in bestList:
    print(item)

# BEST PERFORMANSLAR BÜTÜN DALLARDA
bests=[s for s in listCategories if "Best Performance by an" in s]
bests
winnerFilms=awards[(awards.win==True)&(awards.category=='Best Motion Picture - Drama')]
winnerFilms

# winner best performance

winnerPerformances=awards[(awards.win==True)&((awards.category=='Best Performance by an Actress in a Motion Picture - Drama')|(awards.category=='Best Performance by an Actor in a Motion Picture - Drama'))]
winnerPerformances.tail(50)
winnerPerformances

#seneVeNominee=winnerPerformances[["year_award","nominee"]]
#seneVeNominee
for film in winnerFilms.nominee:
    for performance in winnerPerformances.film:
        if film==performance:
            print(film,"-- >",winnerPerformances.nominee)
            #print(winnerPerformances.nominee)
