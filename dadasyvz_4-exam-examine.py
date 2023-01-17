import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



df=pd.ExcelFile('/kaggle/input/quiz1-1.xls')  #sheet_names metodunu kullanmak icin ilk okuma ExcelFile metodu ile yapildi



sheet_df = []                       #dosya icindeki sheetlerin isimlerinin tutuldugu liste

sheet_df_y = []                     #data frame haline gelecek sheet_df listesindeki isimlerin tekrar kullanilabilmesi icin

total_sheetname_df = []             #sheet_df icindeki elemanlarin dataframe hallerinin guncellenmis hali



[sheet_df.append(i) for i in df.sheet_names]    #dosyadaki sheet isimlerini bulur listeye kaydeder

[sheet_df_y.append(i) for i in sheet_df]        #sheet isimlerini yedekler



for i in range(len(sheet_df)):                                              #sheetleri data frame haline guncelleyerek degistirir

    sheet_df[i] = pd.read_excel("/kaggle/input/quiz1-1.xls", sheet_name=str(sheet_df[i]))

    sheet_df[i]['sinif'] = [sheet_df_y[i] for j in range(len(sheet_df[i]))]  #sheetler data frame haline gelirken sinif adli bir sutunda ekleniyor

    total_sheetname_df.append(sheet_df[i])



all_df=pd.concat([i for i in total_sheetname_df],sort=False)               #tum sheetleri tek dataframe haline getirir                            

all_df.index=np.arange(1,len(all_df)+1)             #all_df frame sabit index atanmasi yapildi

all_df.columns=['isim','dogru','yanlis','bos','sinif']      #dataframe sutun isimleri degistirildi

all_df.replace(to_replace='girmedi',value=0,inplace=True)   #'girmedi' kolonlari 0 ile degistirildi

all_df=all_df.fillna(0)                                     #'nan' kolonlari 0 ile degistirildi

print(all_df)
xls_py_mind = pd.ExcelFile('/kaggle/input/py_mind.xls')

xls_py_science = pd.ExcelFile('/kaggle/input/py_science.xls')

xls_py_sense = pd.ExcelFile('/kaggle/input/py_sense.xls')

xls_py_opinion = pd.ExcelFile('/kaggle/input/py_opinion.xls')
outerIndex_sinif_ismi=["py_mind", "py_mind", "py_mind", "py_mind","py_mind", "py_mind","py_mind", "py_mind","py_mind", "py_mind",\

                       "py_opinion","py_opinion","py_opinion","py_opinion","py_opinion","py_opinion","py_opinion","py_opinion","py_opinion","py_opinion",\

                       "py_science","py_science","py_science","py_science","py_science","py_science","py_science","py_science",\

                       "py_sense","py_sense","py_sense","py_sense","py_sense","py_sense","py_sense","py_sense","py_sense","py_sense","cevap_A."]
sheet_to_df_map = {}

for sheet_name in xls_py_mind.sheet_names:

    sheet_to_df_map[sheet_name] = xls_py_mind.parse(sheet_name)

for sheet_name in xls_py_opinion.sheet_names:

        sheet_to_df_map[sheet_name] = xls_py_opinion.parse(sheet_name)

for sheet_name in xls_py_science.sheet_names:

        sheet_to_df_map[sheet_name] = xls_py_science.parse(sheet_name)

for sheet_name in xls_py_sense.sheet_names:

        sheet_to_df_map[sheet_name] = xls_py_sense.parse(sheet_name)

degerler=list(sheet_to_df_map.values())
InnerIndex_isimler=list(sheet_to_df_map.keys())

InnerIndex_isimler.append("Cevap A.")

InnerIndex_isimler.remove("Blad9")

InnerIndex_isimler.remove("Blad11")

hierarch1=list(zip(outerIndex_sinif_ismi,InnerIndex_isimler))

hierarch1=pd.MultiIndex.from_tuples(hierarch1)
sinavsonuclari=[]

kolonlar=[]

for i in range(len(degerler)):

    kolonlar.append(degerler[i].columns)

cevap_anahtari=[]

for i in range(len(degerler)):

        if len(kolonlar[i])==3:

            sinavsonuclari.append( list(degerler[i][kolonlar[0][2]])[:20])

            cevap_anahtari.append( list(degerler[i][kolonlar[0][1]])[:20])

            print(list(degerler[i][kolonlar[0][2]])[:20])

        elif len(kolonlar[i])==2 :

            sinavsonuclari.append( list(degerler[i][kolonlar[i][1]])[:20])

sinavsonuclari.append(cevap_anahtari[0])

df1=pd.DataFrame(sinavsonuclari,hierarch1,columns=[i for i in range(20)])
correct=[]

blank=[]

wrong=[]

for i in  range(len(degerler)-1):

    countc=0

    countb=0

    countw=0

    for j in range(20):

        if str(df1.iloc[i,j])==str("nan"):

            countb +=1

        elif df1.iloc[i,j]==df1.xs("cevap_A.")[j][0]:

            countc+=1

        elif df1.iloc[i,j]!=df1.xs("cevap_A.")[j][0]:

            countw+=1

    correct.append(countc)

    blank.append(countb)

    wrong.append(countw)

df1["True"]=correct

df1["Wrong"]=wrong

df1["Blank"]=blank



print(df1)
df_quiz3=pd.read_excel("/kaggle/input/quiz-3.xls")

df_quiz3
df_uygulama=pd.read_excel("/kaggle/input/uygulamali1.xls")

df_uygulama
#last_df isimli son dataframe de tum sinavlarin dogru degerini veren sayisal verileri tutulacak

last_df=all_df[["isim","dogru"]]

last_df.columns=["isim","quiz1"]

last_df
quiz2={}                                        #quiz 2 ye girenler quiz1 e girenlerin isim karsiliklari baz

for i in last_df["isim"]:                       #alinarak sozluge atandi

    for j in InnerIndex_isimler:

        if j in i:

            quiz2[i]=df1.xs(j,level=1)["True"][0]

liste=[]

liste1=[]

for j in InnerIndex_isimler:                    #sinava girmemis olanlar tespit edilip quiz2 dicte sonuclari sifir

    liste.append(j)                             #olacak sekilde atandi

for i in last_df["isim"]:

    liste1.append(i)

for i in list(set(liste1).difference(set(liste))):

    quiz2[i] = 0



last_df["quiz2"]=[j for i in last_df["isim"] for k,j in quiz2.items() if i==k]

last_df
quiz3={}                                        #quiz 3 ye girenler quiz1 e girenlerin isim karsiliklari baz

for i in range(1,len(last_df["isim"])+1):                       #alinarak sozluge atandi

   for j in range(len(df_quiz3["name"])):

       if last_df.loc[i,"isim"]== df_quiz3.loc[j,"name"]:

           quiz3[last_df.loc[i,"isim"]]=df_quiz3.loc[j,"true"]

liste=[]

liste1=[]

for j in df_quiz3["name"]:                    #sinava girmemis olanlar tespit edilip quiz2 dicte sonuclari sifir

   liste.append(j)                             #olacak sekilde atandi

for i in last_df["isim"]:

   liste1.append(i)

for i in list(set(liste1).difference(set(liste))):

   quiz3[i] = 0

last_df["quiz3"]=[j for i in last_df["isim"] for k,j in quiz3.items() if i==k]

last_df
uygulamali={}                                        #uygulama sinavina girenler quiz1 e girenlerin isim karsiliklari baz

for i in range(1,len(last_df["isim"])+1):                       #alinarak sozluge atandi

    for j in range(len(df_uygulama["isim"])):

        if last_df.loc[i,"isim"]== df_uygulama.loc[j,"isim"]:

            uygulamali[last_df.loc[i,"isim"]]=df_uygulama.loc[j,"total"]

liste=[]

liste1=[]

for j in df_uygulama["isim"]:                    #sinava girmemis olanlar tespit edilip quiz2 dicte sonuclari sifir

    liste.append(j)                             #olacak sekilde atandi

for i in last_df["isim"]:

    liste1.append(i)

for i in list(set(liste1).difference(set(liste))):

    uygulamali[i] = 0

last_df["uygulamali"]=[j for i in last_df["isim"] for k,j in uygulamali.items() if i==k]

last_df

#siniflarin oldugu sutun ekleniyor

last_df["sinif"]=all_df["sinif"]

last_df
last_df['quiz1'] = last_df['quiz1'].astype('int64')

last_df['quiz2'] = last_df['quiz2'].astype('int64')

last_df['quiz3'] = last_df['quiz3'].astype('int64')

last_df['uygulamali'] = last_df['uygulamali'].astype('int64')
plt.hist(last_df.quiz1,bins=70)

plt.hist(last_df.quiz2,bins=70)

plt.xlabel("py_sense bos dagilimlar")

plt.ylabel("frekans")

plt.title("hist")

plt.show()
plt.bar(last_df.quiz1,last_df.quiz2)

plt.xlabel("py_mind dogru dagilimlar")

plt.ylabel("py_mind yanlis  dagilimlar")

plt.title("bar")

plt.show()
sns.barplot(x=last_df.groupby['quiz1'],y=last_df.groupby['quiz1'])

plt.xticks(rotation=45)

plt.show()