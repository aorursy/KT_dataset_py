import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



df=pd.ExcelFile('../input/quiz1-1.xls')  #sheet_names metodunu kullanmak icin ilk okuma ExcelFile metodu ile yapildi



sheet_df = []                       #dosya icindeki sheetlerin isimlerinin tutuldugu liste

sheet_df_y = []                     #data frame haline gelecek sheet_df listesindeki isimlerin tekrar kullanilabilmesi icin

total_sheetname_df = []             #sheet_df icindeki elemanlarin dataframe hallerinin guncellenmis hali



[sheet_df.append(i) for i in df.sheet_names]    #dosyadaki sheet isimlerini bulur listeye kaydeder

[sheet_df_y.append(i) for i in sheet_df]        #sheet isimlerini yedekler



for i in range(len(sheet_df)):                                              #sheetleri data frame haline guncelleyerek degistirir

    sheet_df[i] = pd.read_excel("../input/quiz1-1.xls", sheet_name=str(sheet_df[i]))

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
rate=[]

for i in range(41):

    rate.append(100*((last_df.loc[i+1,"quiz1"]/40)*0.25 +(last_df.loc[i+1,"quiz2"]/20)*0.25+(last_df.loc[i+1,"quiz3"]/40)*0.25+(last_df.loc[i+1,"uygulamali"]/21)*0.25) )

last_df["succes_rate"]=rate



quiz_result_rate=[]

application_result_rate=[]

for i in range(41):

    quiz_result_rate.append(100*((last_df.loc[i+1,"quiz1"]/40)*0.33 +(last_df.loc[i+1,"quiz2"]/20)*0.33+(last_df.loc[i+1,"quiz3"]/40)*0.34 ))

last_df["quiz_succes_rate"]=quiz_result_rate



for i in range(41):

    application_result_rate.append((last_df.loc[i+1,"uygulamali"]/21)*100 )

last_df["application_succes_rate"]=application_result_rate




sns.barplot(x=last_df.groupby('sinif')['succes_rate'].mean().index,y=last_df.groupby('sinif')['succes_rate'].mean().values)

plt.title("total succes rate of classes")



plt.ylabel("Succes Rate")

plt.xlabel("Class Names")

plt.xticks(rotation=45)

plt.show()
py_sense_rate=last_df[last_df["sinif"]=="py_sense"]

py_opinion_rate=last_df[last_df["sinif"]=="py_opinion"]

py_science_rate=last_df[last_df["sinif"]=="py_science"]

py_mind_rate=last_df[last_df["sinif"]=="py_mind"]

plt.figure(figsize=(15,15))



plt.subplot(4,4,1)

plt.bar(py_sense_rate.isim,py_sense_rate.succes_rate,color="red")

plt.title("py sense student success rate")

plt.xticks(rotation=90)





plt.subplot(4,4,2)



plt.bar(py_opinion_rate.isim,py_opinion_rate.succes_rate,color="blue")

plt.xticks(rotation=90)

plt.title("py opinion student success rate")





plt.subplot(4,4,3)



plt.bar(py_science_rate.isim,py_science_rate.succes_rate,color="yellow")

plt.xticks(rotation=90)

plt.title("py science student success rate")





plt.subplot(4,4,4)



plt.bar(py_mind_rate.isim,py_mind_rate.succes_rate,color="green")

plt.xticks(rotation=90)

plt.title("py mind student success rate")



plt.show()



plt.figure(figsize=(20,15))

plt.subplot(4,4,1)

plt.plot(py_sense_rate.isim,py_sense_rate.succes_rate,color="red",label="py_sense")

plt.title("py sense student success rate")

plt.legend() # x ve ye koordinatlarini olusturuyo

plt.xticks(rotation=30)

plt.subplot(4,4,2)

plt.plot(py_opinion_rate.isim,py_opinion_rate.succes_rate,color="blue",label="py_opinion")

plt.title("py opinion student success rate")

plt.legend() # x ve ye koordinatlarini olusturuyo

plt.xticks(rotation=30)

plt.subplot(4,4,3)



plt.plot(py_science_rate.isim,py_science_rate.succes_rate,color="yellow",label="py_science")

plt.title("py science student success rate")

plt.legend() # x ve ye koordinatlarini olusturuyo

plt.xticks(rotation=30)

plt.subplot(4,4,4)



plt.plot(py_mind_rate.isim,py_mind_rate.succes_rate,color="green",label="py_mind")

plt.title("py mind student success rate")







plt.legend() # x ve ye koordinatlarini olusturuyo

plt.xticks(rotation=30)



plt.xlabel("names")

plt.ylabel("succes rate")

plt.show()



color_dict = {'py_science':'green', 'py_mind':'red','py_sense':'blue', 'py_opinion':'orange'}

plt.bar(last_df.groupby('sinif')['quiz_succes_rate'].mean().index,last_df.groupby('sinif')['quiz_succes_rate'].mean().values,color=[color_dict[r] for r in list(color_dict)])

plt.title("total success rate of three test quiz")

plt.ylabel("quiz_result_rate")

plt.xlabel("Class Names")

plt.xticks(rotation=45)

plt.show()



plt.bar(last_df.groupby('sinif')['application_succes_rate'].mean().index,last_df.groupby('sinif')['application_succes_rate'].mean().values,color=[color_dict[r] for r in list(color_dict)])

plt.title("total success rate of application quiz")



plt.ylabel("application result rate")

plt.xlabel("Class Names")

plt.xticks(rotation=45)

plt.show()
last_df["quiz1-Rate"]=last_df["quiz1"]/40*100

last_df["quiz2-Rate"]=last_df["quiz2"]/20*100

last_df["quiz3-Rate"]=last_df["quiz3"]/40*100

last_df["Application-Rate"]=last_df["uygulamali"]/21*100

last_df.head()
first_5 = last_df.sort_values(by=['succes_rate'], ascending=False)[0:5]

grades=first_5.loc[:,"quiz1-Rate":"Application-Rate"]

grades["isim"]=first_5["isim"]

grades.set_index ("isim",inplace=True)

grades = grades.transpose()

grades.columns=['Dilek_Koksal', 'Sehri_Gokcan', 'Ufuk_Doymaz', 'Ilhami',

       'Hakan_Yildirim']

plt.figure(figsize=(8,8))

plt.plot(grades.index,grades.Dilek_Koksal,color="red")

plt.plot(grades.index,grades.Sehri_Gokcan,color="yellow")

plt.plot(grades.index,grades.Ufuk_Doymaz,color="green")

plt.plot(grades.index,grades.Ilhami,color="blue")

plt.plot(grades.index,grades.Hakan_Yildirim,color="orange")

plt.title("Firs five students' success chart")

plt.legend() # x ve ye koordinatlarini olusturuyo

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10,10))

plt.bar(last_df.sort_values(by=['succes_rate'], ascending=False).isim,last_df.sort_values(by=['succes_rate'], ascending=False).succes_rate,color="orange")

plt.title("Firs five students' success chart")

plt.xticks(rotation=90)

plt.show()