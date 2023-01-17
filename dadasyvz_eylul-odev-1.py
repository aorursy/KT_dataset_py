import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



df=pd.ExcelFile('../input/examresult/quiz1-1.xlsx')  #sheet_names metodunu kullanmak icin ilk okuma ExcelFile metodu ile yapildi



sheet_df = []                       #dosya icindeki sheetlerin isimlerinin tutuldugu liste

sheet_df_y = []                     #data frame haline gelecek sheet_df listesindeki isimlerin tekrar kullanilabilmesi icin

total_sheetname_df = []             #sheet_df icindeki elemanlarin dataframe hallerinin guncellenmis hali



[sheet_df.append(i) for i in df.sheet_names]    #dosyadaki sheet isimlerini bulur listeye kaydeder

[sheet_df_y.append(i) for i in sheet_df]        #sheet isimlerini yedekler



for i in range(len(sheet_df)):                                              #sheetleri data frame haline guncelleyerek degistirir

    sheet_df[i] = pd.read_excel("../input/examresult/quiz1-1.xlsx", sheet_name=str(sheet_df[i]))

    sheet_df[i]['sinif'] = [sheet_df_y[i] for j in range(len(sheet_df[i]))]  #sheetler data frame haline gelirken sinif adli bir sutunda ekleniyor

    total_sheetname_df.append(sheet_df[i])



all_df=pd.concat([i for i in total_sheetname_df],sort=False)               #tum sheetleri tek dataframe haline getirir                            

all_df.index=np.arange(1,len(all_df)+1)             #all_df frame sabit index atanmasi yapildi

all_df.columns=['isim','dogru','yanlis','bos','sinif']      #dataframe sutun isimleri degistirildi

all_df.replace(to_replace='girmedi',value=0,inplace=True)   #'girmedi' kolonlari 0 ile degistirildi

all_df=all_df.fillna(0)                                     #'nan' kolonlari 0 ile degistirildi

print(all_df)
#sinava katilan ogrenci sayisi

count=0

for i in range(len(all_df)):                       #dogrusu olmayan ogrencini yanlis ve bosuda yok ise sinava katilmadigi varsayildi

    if all_df.iloc[i]['dogru']==0:

        if  all_df.iloc[i]['yanlis']==0:

            if all_df.iloc[i]['bos']==0:

                count+=1

sinava_katilan= len(all_df)-count           

print('tum siniflardan sinava katilan=',sinava_katilan)



#py_sense sinava katilan py_sense_SK

count=0

for i in range(len(all_df[all_df["sinif"]=="py_sense"])):

    if all_df[all_df["sinif"]=="py_sense"].iloc[i]['dogru']==0:

        if  all_df[all_df["sinif"]=="py_sense"].iloc[i]['yanlis']==0:

            if all_df[all_df["sinif"]=="py_sense"].iloc[i]['bos']==0:

                count+=1

py_sense_SK=len(all_df[all_df["sinif"]=="py_sense"])-count

print('py_sense sinava katilan=',py_sense_SK)



#py_science sinava katilan

count=0

for i in range(len(all_df[all_df["sinif"]=="py_science"])):

    if all_df[all_df["sinif"]=="py_science"].iloc[i]['dogru']==0:

        if  all_df[all_df["sinif"]=="py_science"].iloc[i]['yanlis']==0:

            if all_df[all_df["sinif"]=="py_science"].iloc[i]['bos']==0:

                count+=1

py_science_SK=len(all_df[all_df["sinif"]=="py_science"])-count

print('py_science sinava katilan=',py_science_SK)



#py_opinion sinava katilan

count=0

for i in range(len(all_df[all_df["sinif"]=="py_opinion"])):

    if all_df[all_df["sinif"]=="py_opinion"].iloc[i]['dogru']==0:

        if  all_df[all_df["sinif"]=="py_opinion"].iloc[i]['yanlis']==0:

            if all_df[all_df["sinif"]=="py_opinion"].iloc[i]['bos']==0:

                count+=1

py_opinion_SK=len(all_df[all_df["sinif"]=="py_opinion"])-count

print('py_opinion sinava katilan=',py_opinion_SK)



#py_mind sinava katilan

count=0

for i in range(len(all_df[all_df["sinif"]=="py_mind"])):

    if all_df[all_df["sinif"]=="py_mind"].iloc[i]['dogru']==0:

        if  all_df[all_df["sinif"]=="py_mind"].iloc[i]['yanlis']==0:

            if all_df[all_df["sinif"]=="py_mind"].iloc[i]['bos']==0:

                count+=1

py_mind_SK=len(all_df[all_df["sinif"]=="py_mind"])-count

print('py_mind sinava katilan=',py_mind_SK)
#sinavdaki dogru ortalamasi

print('tum siniflarin dogru ortalamasi',all_df.dogru.sum()/sinava_katilan)



#sinif bazinda dogru ortalamalari

py_sense_DO=(all_df[all_df["sinif"]=="py_sense"].dogru.sum())/py_sense_SK

py_science_DO=(all_df[all_df["sinif"]=="py_science"].dogru.sum())/py_science_SK

py_opinion_DO=(all_df[all_df["sinif"]=="py_opinion"].dogru.sum())/py_opinion_SK

py_mind_DO=(all_df[all_df["sinif"]=="py_mind"].dogru.sum())/py_mind_SK

print('py_sense sinifi dogru ortalama=',py_sense_DO)

print('py_science sinifi dogru ortalama=',py_science_DO)

print('py_opinion sinifi dogru ortalama=',py_opinion_DO)

print('py_mind sinifi dogru ortalama=',py_mind_DO)
#en basarili sinif

if py_mind_DO>py_sense_DO:

    print('sinavdaki en basarili sinif=py_mind')

elif py_opinion_DO>py_mind_DO:

    print('sinavdaki en basarili sinif=py_opinion')

elif py_science_DO>py_opinion_DO:

    print('sinavdaki en basarili sinif=py_science')

else:

    print('sinavdaki en basarili sinif=py_sense')
#butun siniflar icinde en basarili 3 kisi

print(all_df.sort_values("dogru", ascending=False).nlargest(3, "dogru"))
#sinif bazinda en basarili kisiler

print('py_sense sinifi en basarili kisisi=\n',all_df[all_df["sinif"]=="py_sense"].sort_values("dogru", ascending=False).nlargest(1, "dogru"))

print('py_science sinifi en basarili kisisi=\n',all_df[all_df["sinif"]=="py_science"].sort_values("dogru", ascending=False).nlargest(1, "dogru"))

print('py_opinion sinifi en basarili kisisi=\n',all_df[all_df["sinif"]=="py_opinion"].sort_values("dogru", ascending=False).nlargest(1, "dogru"))

print('py_mind sinifi en basarili kisisi=\n',all_df[all_df["sinif"]=="py_mind"].sort_values("dogru", ascending=False).nlargest(1, "dogru"))
#siniflarin dogru cevap dagilimlarina gore plot grafigini verir"

py_sense=all_df[all_df.sinif=="py_sense"]

py_science=all_df[all_df.sinif=="py_science"]

py_opinion=all_df[all_df.sinif=="py_opinion"]

py_mind=all_df[all_df.sinif=="py_mind"]

plt.plot(py_sense.sinif,py_sense.dogru,color="red",label= "py_sense")

plt.plot(py_science.sinif,py_science.dogru,color="red",label= "py_science")

plt.plot(py_opinion.sinif,py_opinion.dogru,color="red",label= "py_opinion")

plt.plot(py_mind.sinif,py_mind.dogru,color="red",label= "py_mind")

plt.show()
#siniflarin yanlis cevap dagilimlarina gore scatter grafigini verir

plt.scatter(py_sense.sinif, py_sense.yanlis, color="red", label="py_sense")

plt.scatter(py_opinion.sinif, py_opinion.yanlis, color="blue", label="py_opinion")

plt.scatter(py_science.sinif, py_science.yanlis, color="black", label="py_science")

plt.scatter(py_mind.sinif, py_mind.yanlis, color="yellow", label="py_mind")

plt.show()
#siniflarin bos cevap dagilimlarina gore histogram grafiklerini verir

plt.hist(py_sense.bos, bins=70)

plt.xlabel("py_sense bos dagilimlar")

plt.ylabel("frekans")

plt.title("hist")

plt.show()

plt.hist(py_opinion.bos, bins=70)

plt.xlabel("py_opinion bos dagilimlar")

plt.ylabel("frekans")

plt.title("hist")

plt.show()

plt.hist(py_science.bos, bins=70)

plt.xlabel("py_science bos dagilimlar")

plt.ylabel("frekans")

plt.title("hist")

plt.show()

plt.hist(py_mind.bos, bins=70)

plt.xlabel("py_mind bos dagilimlar")

plt.ylabel("frekans")

plt.title("hist")

plt.show()

#siniflarin dogru yanlis cevap dagilimlarina gore bar grafiklerini verir

plt.bar(py_sense.dogru,py_sense.yanlis)

plt.xlabel("py_sense dogru dagilimlar")

plt.ylabel("py_sense yanlis dagilimlar")

plt.title("bar")

plt.show()

plt.bar(py_opinion.dogru, py_opinion.yanlis)

plt.xlabel("py_opinion dogru dagilimlar")

plt.ylabel("py_opinion yanlis  dagilimlar")

plt.title("bar")

plt.show()

plt.bar(py_science.dogru, py_science.yanlis)

plt.xlabel("py_science dogru dagilimlar")

plt.ylabel("py_science yanlis  dagilimlar")

plt.title("bar")

plt.show()

plt.bar(py_mind.dogru, py_mind.yanlis)

plt.xlabel("py_mind dogru dagilimlar")

plt.ylabel("py_mind yanlis  dagilimlar")

plt.title("bar")

plt.show()
xls_py_mind = pd.ExcelFile('/kaggle/input/examresult/py_mind.xlsx')

xls_py_science = pd.ExcelFile('/kaggle/input/examresult/py_science.xlsx')

xls_py_sense = pd.ExcelFile('/kaggle/input/examresult/py_sense.xlsx')

xls_py_opinion = pd.ExcelFile('/kaggle/input/examresult/py_opinion.xlsx')
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
print("The number of people Who attend the second Quiz:{}".format(len(InnerIndex_isimler)-1))
print("The most 3 succesful students according to their True answer numbers\n",df1.sort_values("True",ascending=False).nlargest(4,"True")["True"][1:4])
means={}

means["py_mind"]=df1.loc["py_mind"].mean()

means["Py_opinion"]=df1.loc["py_opinion"].mean()

means["py_science"]=df1.loc["py_science"].mean()

means["Py_sense"]=(df1.loc["py_sense"].mean())

print("Mean of the Classes\n",pd.DataFrame(means))
print("the most succesfull class according to their true answers\n",pd.DataFrame(means).T.sort_values("True",ascending=False).nlargest(1,"True"))
print("the most succesful student of the every class\n")

print("Py Mind:",df1.loc["py_mind"].sort_values("True",ascending=False).nlargest(1, "True")["True"])

print("Py Opinion:",df1.loc["py_opinion"].sort_values("True",ascending=False).nlargest(1, "True")["True"])

print("Py science:",df1.loc["py_science"].sort_values("True",ascending=False).nlargest(1, "True")["True"])

print("Py sense:",df1.loc["py_sense"].sort_values("True",ascending=False).nlargest(1, "True")["True"])