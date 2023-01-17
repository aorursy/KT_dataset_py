import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Stackoverflow'un 2019 yılı için geliştiricilerle yaptığı anket sonuçlarını içeren "survey_results_public.csv" adlı dosyayı okuyarak survey_19 adlı DataFrame'e aktarın.
survey_19 = pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv");
survey_19.head()
# DataFrame'in satır ve sütun sayısını yazdırın.
survey_19.shape
# Respondent sütununu index olarak atayın.
survey_19.set_index("Respondent", inplace=True)
survey_19.head()
# Aşağıdaki sütunları dataframe'den siliniz:
# ['ScreenName', 'SOVisit1st','SOVisitFreq', 'SOVisitTo', 'SOFindAnswer', 'SOTimeSaved', 'SOHowMuchTime', 'SOAccount',   'SOPartFreq', 'SOJobs', 'EntTeams', 'SOComm', 'WelcomeChange', 'SONewContent']
survey_19.drop(['ScreenName', 'SOVisit1st','SOVisitFreq', 'SOVisitTo', 'SOFindAnswer', 'SOTimeSaved', 'SOHowMuchTime', 'SOAccount',   'SOPartFreq', 'SOJobs', 'EntTeams', 'SOComm', 'WelcomeChange', 'SONewContent'], axis=1, inplace=True)
survey_19.head()
survey_19.isna().groupby(['DevType','LanguageWorkedWith']).size()
# Kaç satır silindi?
survey_19.isna().groupby(['DevType','LanguageWorkedWith']).size().iloc[[1,2,3]].sum()
# DevType ya da LanguageWorkedWith sütununda eksik veri olan satırları siliniz.
survey_19.dropna(subset=["DevType", "LanguageWorkedWith"], inplace=True)
survey_19.head()
# Country sütunundaki Russian Federation ve Czech Republic değerlerini sırasıyla Russia ve Czechia olarak değiştiriniz.
survey_19[survey_19["Country"] == "Russian Federation"].head()
survey_19["Country"] = survey_19["Country"].apply(lambda x: "Russia" if x == "Russian Federation" else x)
survey_19[survey_19["Country"] == "Russia"].head()
survey_19[survey_19["Country"] == "Czech Republic"].head()
survey_19["Country"] = survey_19["Country"].apply(lambda x: "Czechia" if x == "Czech Republic" else x)
survey_19[survey_19["Country"] == "Czechia"].head()
survey_19.shape
# Haftalık çalışma saati (WorkWeekHrs) ya da yıllık maaş (ConvertedComp) sütunlarındaki değerleri, ilgili sütunun ortalamasının %99'undan daha uzakta veri içeren satırları siliniz.
mean99WWH = survey_19["WorkWeekHrs"].mean()
mean99WWHUpper = mean99WWH + (mean99WWH * 99 / 100)
mean99WWHLower = mean99WWH - (mean99WWH * 99 / 100)
mean99CC = survey_19["ConvertedComp"].mean()
mean99CCUpper = mean99CC + (mean99CC * 99 / 100)
mean99CCLower = mean99CC - (mean99CC * 99 / 100)
survey_19.drop(survey_19[(survey_19["WorkWeekHrs"] > mean99WWHUpper) | (survey_19["WorkWeekHrs"] < mean99WWHLower) | (survey_19["WorkWeekHrs"].isna())]["WorkWeekHrs"].index, inplace=True)
survey_19.drop(survey_19[(survey_19["ConvertedComp"] > mean99CCUpper) | (survey_19["ConvertedComp"] < mean99CCLower) | (survey_19["ConvertedComp"].isna())]["ConvertedComp"].index, inplace=True)
survey_19.shape
# Gender sütununda Man, Woman ve NaN dışındaki değerleri, Non-binary değeri ile değiştirin.
survey_19["Gender"] = survey_19["Gender"].apply(lambda x: "Non-binary" if x not in ["Man", "Woman", "NaN"] else x)
survey_19["Gender"]
# Age sütununu 0, 15, 19, 24, 29, 34, 39, 44, 49, 54, 59, 99 binleri ile aralığa çevirerek AgeInterval adında yeni bir sütun oluşturunuz.
survey_19["AgeInterval"] = pd.cut(survey_19["Age"], bins=[0, 15, 19, 24, 29, 34, 39, 44, 49, 54, 59, 99], include_lowest=True)
survey_19["AgeInterval"]
# Ankete katılanların yaş aralıklarına göre dağılımını pasta grafiğinde gösteriniz.
plt.figure(figsize=(20,8))
plt.pie(survey_19["AgeInterval"].value_counts()[survey_19["AgeInterval"].value_counts() > 0], labels=survey_19["AgeInterval"].unique().dropna())
pd.Categorical(survey_19['AgeInterval']).unique()
# (19-24] yaş aralığının en çok kullandığı sosyal medya nedir?
survey_19[survey_19["AgeInterval"] == pd.Categorical(survey_19['AgeInterval']).unique()[1]]["SocialMedia"].value_counts().head(1)
plt.figure(figsize=(20,8))
survey_19.groupby("CareerSat").mean()["ConvertedComp"].loc[["Very dissatisfied","Slightly dissatisfied","Neither satisfied nor dissatisfied","Slightly satisfied","Very satisfied"]].plot()
# Her bir DevType tipinin sütun olarak, ankete katılanların da satır olarak temsil edildiği yeni bir DataFrame oluşturup devtype_df adlı değişkene atayın.
devtype_df = pd.DataFrame(columns=["DevType"], data=survey_19["DevType"])
devtype_df
# Oluşan devtype_df DataFrame'ine Gender sütununu ekleyin.
devtype_df["Gender"] = survey_19["Gender"]
devtype_df
# ankete katılan kadın-erkek oranından daha yüksek orana sahip
perc = devtype_df[devtype_df["Gender"] == "Woman"]["Gender"].value_counts()[0] * 100 / devtype_df.shape[0]
perc
# DataFrame'i cinsiyete göre gruplayarak her bir DevType için cinsiyete göre sayıları bulun.
dfMWN = devtype_df.groupby("DevType").count()["Gender"].to_frame()

dfW = devtype_df[devtype_df["Gender"] == "Woman"]
dfWDev = dfW.groupby("DevType").count()["Gender"].to_frame()

dfM = devtype_df[devtype_df["Gender"] == "Man"]
dfMDev = dfM.groupby("DevType").count()["Gender"].to_frame()

dfN = devtype_df[devtype_df["Gender"] == "Non-binary"]
dfNDev = dfN.groupby("DevType").count()["Gender"].to_frame()

dfMWN = dfMWN.join(dfWDev.rename(columns={"Gender": "Woman"}))
dfMWN = dfMWN.join(dfMDev.rename(columns={"Gender": "Man"}))
dfMWN = dfMWN.join(dfNDev.rename(columns={"Gender": "Non-binary"}))

dfMWN
# DevType tiplerindeki kadın-erkek oranlarını, ankete katılan kadın-erkek oranı ile karşılaştırarak kadınların daha çok tercih ettiği/edildiği DevType tiplerini listeleyin.
dfMWN["Perc"] = dfMWN["Woman"] * 100 / dfMWN["Gender"]
dfMWN[dfMWN["Perc"] > perc].sort_values("Perc", ascending=False)
fig = plt.figure(figsize=(20,8))
ax1 = plt.subplot(1,2,1)
plt.pie(survey_19["Country"].value_counts().head(10), labels=survey_19["Country"].value_counts().head(10).index)
ax1 = plt.subplot(1,2,2)
plt.pie(survey_19["Country"].value_counts().tail(10), labels=survey_19["Country"].value_counts().tail(10).index)
# Veri Bilimcilerin (DevType sütununda Data scientist or machine learning specialist değerini içeren) en çok kullandığı dil (LanguageWorkedWith) nedir?
survey_19[survey_19["DevType"].apply(lambda x: "Data scientist or machine learning specialist" in x)]["LanguageWorkedWith"].value_counts().to_frame().head(1)
# Tüm kullandıkları dillerin yüzdelerini bulunuz ve yatay bar plot olarak gösteriniz.
plt.figure(figsize=(5,600))
survey_19[survey_19["DevType"].apply(lambda x: "Data scientist or machine learning specialist" in x)]["LanguageWorkedWith"].value_counts().plot.barh()
# Haftalık çalışma saatlerinin (WorkWeekHrs) ülke bazlı ortalama, standart sapma ve medyan değerlerinden oluşan bir dataframe oluşturunuz.
survey_19.groupby("Country").agg(["mean", "std", "median"])["WorkWeekHrs"]
# Ortalamaya göre büyükten küçüğe sıralayınız.
survey_19.groupby("Country").agg(["mean", "std", "median"])["WorkWeekHrs"].sort_values("mean", ascending=False)
# Türkiye'nin kaçıncı sırada olduğunu ve haftalık çalışma saatini bulunuz.
dfTemp = survey_19.groupby("Country").agg(["mean", "std", "median"])["WorkWeekHrs"].reset_index()
dfTemp[dfTemp["Country"] == "Turkey"]
plt.figure(figsize=(20,8))
survey_19[survey_19["CodeRevHrs"] < 40]["CodeRevHrs"].plot.hist(bins=20)
nameChn = {'I am a student who is learning to code': 'Student',
  'I am not primarily a developer, but I write code sometimes as part of my work': 'Not a Developer',
  'I am a developer by profession':  'Developer',
  'I code primarily as a hobby':   'Code as hobby',
  'I used to be a developer by profession, but no longer am': 'Was a Developer'}
survey_19["MainBranch"] = survey_19["MainBranch"].apply(lambda x: list(nameChn.values())[(list(nameChn.keys()).index(x))])
survey_19["MainBranch"]
# Ana branş (MainBranch) X-ekseninde olacak şekilde her bir branştakilerin sayısını bar plot ile gösteriniz. Her bir bar, Cinsiyet (Gender) sayısına göre yığıt (stacked) şeklinde gösterilmelidir.
plt.figure(figsize=(20,8))
survey_19[survey_19["Gender"] == "Man"].groupby("MainBranch").count()["Gender"].plot(kind="bar", stacked=True, color='r')
survey_19[survey_19["Gender"] == "Woman"].groupby("MainBranch").count()["Gender"].plot(kind="bar", stacked=True, color='g')
survey_19[survey_19["Gender"] == "Non-binary"].groupby("MainBranch").count()["Gender"].plot(kind="bar", stacked=True, color='b')
# Ana branşı Developer olan ve Türkiye'den ankete katılanların eğitim seviyelerine göre dağılımını pasta grafiği ile gösteriniz.
plt.figure(figsize=(20,8))
survey_19[(survey_19["MainBranch"].str.contains("Developer")) & (survey_19["Country"] == "Turkey")]["EdLevel"].value_counts().plot.pie()
# Türkiye'den ankete katılan ve TL cinsinden maaş alanların ortalama yıllık brüt maaşını bulunuz.
# Her üç sütundan herhangi biri NaN olan satırları ihmal ediniz.
# Frekansı (CompFreq) haftalık olanlar için bir yılı 50 hafta, aylık olanlar için ise 12 ay olarak düşününüz.
dfTR = survey_19[(survey_19["Country"] == "Turkey") & (survey_19["CurrencySymbol"] == "TRY")][["CurrencySymbol", "CompTotal", "CompFreq"]]
dfTR["TotalTotal"] = np.where(dfTR["CompFreq"] == "Weekly", dfTR["CompTotal"] * 50, dfTR["CompTotal"])
dfTR["TotalTotal"] = np.where(dfTR["CompFreq"] == "Monthly", dfTR["CompTotal"] * 12, dfTR["CompTotal"])
dfTR
# Türkiye'den ankete katılan ve TL cinsinden maaş alanların ortalama yıllık brüt maaşını bulunuz.
dfTR["TotalTotal"].mean()
plt.figure(figsize=(20,8))
sns.countplot(x="OpSys", data=survey_19, hue="OpenSourcer")
# Ülkelerdeki eğitim düzeyi (EdLevel) oranlarını içeren country_edlevels adında bir dataframe oluşturunuz. (Satır ülkeler, sütun eğitim düzeyleri olacak şekilde)
temp = survey_19.groupby(["Country", "EdLevel"]).count().reset_index()
country_edlevels = pd.pivot_table(temp, index=["Country"], columns=["EdLevel"])["Age"]
country_edlevels
country_edlevels["sum"] = country_edlevels.sum(axis=1)
country_edlevels = country_edlevels.sort_values("sum", ascending=False)
country_edlevels
# Ankete en çok katılım gösteren 5 ülkeyi, her bir eğitim düzeyi oranı için karşılaştırmalı olarak gösteren bir bar plot çiziniz.
plt.figure(figsize=(20,8))
list0 = survey_19.groupby("Country").count()["EdLevel"].to_frame().sort_values("EdLevel", ascending=False).head(5)
list1 = list0.index.tolist()
list2 = survey_19[survey_19["Country"].isin(list1)]
sns.countplot(x="Country", data=list2, hue="EdLevel")
country_info = pd.read_table("https://download.geonames.org/export/dump/countryInfo.txt", skiprows=49, usecols = ["Country", "Continent", "CurrencyCode"], index_col="Country", keep_default_na=False)
country_info
code = {"AF" : "Africa",
  "AS" : "Asia",           
  "EU" : "Europe",            
  "NA" : "North America",        
  "OC" : "Oceania",            
  "SA" : "South America",        
  "AN" : "Antarctica"}
country_info["ContinentName"] = country_info["Continent"].map(code)
country_info
# survey_19 dataframe'i ile country_info dataframe'ini ülke bazında birleştirip merged_survey isimli yeni bir dataframe'e aktarınız.
merged_survey = pd.merge(survey_19, country_info, left_on="Country", right_on="Country")
merged_survey
# Yaşadığı ülkenin resmi para biriminin dışındaki para birimiyle maaş alan kişi sayısı nedir?
merged_survey[merged_survey["CurrencySymbol"] != merged_survey["CurrencyCode"]].shape[0]
# Her bir para biriminin kaç ülkede maaş olarak verildiğini azalan sırada listeleyiniz.
temp = merged_survey.groupby(["Country", "CurrencySymbol"]).count().reset_index()[["Country","CurrencySymbol"]]
temp.groupby(["CurrencySymbol"]).count().reset_index().sort_values("Country", ascending=False)
# Her kıtadaki ortalama maaşı en yüksek olan ülkeyi ve ortalama maaşı listeleyiniz.
merged_survey["TotalTotal"] = np.where(merged_survey["CompFreq"] == "Weekly", merged_survey["CompTotal"] * 50, merged_survey["CompTotal"])
merged_survey["TotalTotal"] = np.where(merged_survey["CompFreq"] == "Monthly", merged_survey["CompTotal"] * 12, merged_survey["CompTotal"])
temp = merged_survey.groupby(["Continent","Country"])["TotalTotal"].agg(["mean"]).sort_values("mean", ascending=False)
temp2 = temp.groupby(["Continent","Country"])["mean"].sum().to_frame()["mean"].to_frame().reset_index().sort_values("mean", ascending=False)
temp2.groupby(["Continent"]).first()
# Avrupa kıtasındaki ülkelerin ortalama haftalık çalışma saati (WorkWeekHrs) ve ortalama yıllık maaşlarını (ConvertedComp) saçılım (scatter) grafiğinde gösteriniz.
plt.figure(figsize=(20,8))
sns.scatterplot(x="WorkWeekHrs", y="ConvertedComp", data=merged_survey[merged_survey["Continent"] == "EU"])