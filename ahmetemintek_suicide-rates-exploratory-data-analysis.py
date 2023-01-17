import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from scipy.stats.mstats import winsorize

from scipy.stats import stats

from scipy.stats import zscore

from scipy.stats import jarque_bera

from scipy.stats import normaltest

from sklearn.preprocessing import normalize

from sklearn.preprocessing import scale

import warnings



warnings.filterwarnings("ignore")

%matplotlib inline

sns.set(style="whitegrid")

title_font= {"family": "arial", "weight": "bold", "color": "darkred", "size": 13}

label_font= {"family": "arial", "weight": "bold", "color": "darkblue", "size": 10}

df= pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")

df.head() # We only display the first five lines

df.info()
df.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population',

       'suicides/100k_rate', 'country-year', 'HDI_for_year',

       'gdp_for_year', 'gdp_per_capita', 'generation']

df.columns
pd.options.display.float_format= "{:.6f}".format

df.describe()
df.isnull().sum()*100/df.shape[0]
df.nunique()
columns_names = ["year", "sex", "age", "generation", "country"]

for col in columns_names:

    print("{} unique values:\n {}".format(col, df[col].unique()))
df["gdp_for_year"].str.strip()
df["gdp_for_year"] = df["gdp_for_year"].replace(",", "", regex=True)

df["gdp_for_year"]
df["gdp_for_year"]= df["gdp_for_year"].astype("int64")
print("incorrect values for {}: ".format("gdp_for_year"))

for value in df["gdp_for_year"]:

    try:

        float(value)

    except:

        print(value)
df.info()
plt.figure(figsize=(18,10))

columns_name = ["suicides_no", "population","suicides/100k_rate", "gdp_per_capita", "gdp_for_year" ]

for i in range(5):

    plt.subplot(2,3,i+1)

    plt.boxplot(df[columns_name[i]])

    plt.title("{} box graph".format(columns_name[i]), fontdict= title_font)

    
plt.figure(figsize=(18,10))

columns_name = ["suicides_no", "population","suicides/100k_rate", "gdp_per_capita", "gdp_for_year" ]

for i in range(5):

    plt.subplot(2,3,i+1)

    plt.hist(df[columns_name[i]])

    plt.title("{} histogram graph".format(columns_name[i]), fontdict=title_font)

    

    
columns_name = ["suicides_no", "population","suicides/100k_rate", "gdp_per_capita", "gdp_for_year" ]

for names in range(0,5): 

    zscorelist = []

    zscores = zscore(df[columns_name[names]])

    for thereshold in np.arange(0,5,0.1):

        zscorelist.append((thereshold,len(np.where(zscores>thereshold)[0]))) 

        df_outliers= pd.DataFrame(zscorelist, columns=["thereshold", "outliers"])

    plt.plot(df_outliers.thereshold, df_outliers.outliers)

    plt.title(columns_name[names], fontdict=title_font)

    plt.show()

    
columns_name = ["suicides_no", "population","suicides/100k_rate", "gdp_per_capita", "gdp_for_year" ]

for col in columns_name:

    q75, q25 = np.percentile(df[col], [75,25])

    caa = q75-q25

    comparison = pd.DataFrame(columns= [col, "thereshold", "outliers"])

    for thereshold in np.arange(1,5,0.5):

        min_value= q25- (caa*thereshold)

        max_value= q75+ (caa*thereshold)

        outliers= len(np.where((df[col]>max_value) | (df[col]<min_value))[0])

        comparison = comparison.append({col: col, "thereshold": thereshold,

                                             "outliers": outliers }, ignore_index=True)

    display(comparison)

    

        
df["winsorize_suicides_no"] = winsorize(df["suicides_no"], (0, 0.11))

plt.figure(figsize=(12,5))

plt.subplot(121)

plt.boxplot(df["winsorize_suicides_no"], whis=2.5)

plt.title("winsorize suicides_no", fontdict=title_font)



plt.subplot(122)

plt.boxplot(df["suicides_no"])

plt.title("suicides_no", fontdict=title_font)

plt.show()
df["winsorize_suicides/100k_rate"] = winsorize(df["suicides/100k_rate"], (0,0.05))

plt.figure(figsize=(12,5))

plt.subplot(121)

plt.boxplot(df["winsorize_suicides/100k_rate"], whis=2.5)

plt.title("winsorize suicides/100k_rate", fontdict=title_font)



plt.subplot(122)

plt.boxplot(df["suicides/100k_rate"])

plt.title("suicides/100k_rate", fontdict=title_font)

plt.show()
df["winsorize_gdp_per_capita"] = winsorize(df["gdp_per_capita"], (0, 0.03))

plt.figure(figsize=(12,5))

plt.subplot(121)

plt.boxplot(df["winsorize_gdp_per_capita"], whis=2.0)

plt.title("winsorize gdp_per_capita", fontdict=title_font)



plt.subplot(122)

plt.boxplot(df["gdp_per_capita"])

plt.title("gpd_per_capita", fontdict=title_font)

plt.show()
df["winsorize_gdp_for_year"] = winsorize(df["gdp_for_year"], (0, 0.11))

plt.figure(figsize=(12,5))

plt.subplot(121)

plt.boxplot(df["winsorize_gdp_for_year"], whis=2.5)

plt.title("winsorize gdp_for_year", fontdict=title_font)



plt.subplot(122)

plt.boxplot(df["gdp_for_year"])

plt.title("gdp_gor_year", fontdict=title_font)

plt.show()
df["winsorize_population"]= winsorize(df["population"], (0,0.09))

plt.figure(figsize=(12,5))

plt.subplot(121)

plt.boxplot(df["winsorize_population"], whis=3.0)

plt.title("winsorize_population", fontdict=title_font)



plt.subplot(122)

plt.boxplot(df["population"])

plt.title("population", fontdict=title_font)

plt.show()
columns_name = ["suicides_no", "winsorize_suicides_no", "suicides/100k_rate", 

                "winsorize_suicides/100k_rate",  "gdp_per_capita",

                "winsorize_gdp_per_capita","gdp_for_year", "winsorize_gdp_for_year", "population", "winsorize_population" ]

plt.figure(figsize=(30,12))

for i in range(2):

    plt.subplot(5,2,i+1)

    plt.hist(df[columns_name[i]])

    plt.title(columns_name[i], fontdict=title_font)

for i in range(2):

    plt.subplot(5,2,i+3)

    plt.hist(df[columns_name[i+2]])

    plt.title(columns_name[i+2], fontdict=title_font)

for i in range(2):

    plt.subplot(5,2,i+5)

    plt.hist(df[columns_name[i+4]])

    plt.title(columns_name[i+4], fontdict=title_font)

for i in range(2):

    plt.subplot(5,2,i+7)

    plt.hist(df[columns_name[i+6]])

    plt.title(columns_name[i+6], fontdict=title_font)

for i in range(2):

    plt.subplot(5,2,i+9)

    plt.hist(df[columns_name[i+8]])

    plt.title(columns_name[i+8], fontdict=title_font)



plt.show()  

        

    
columns_name= ["suicides_no", "population", "suicides/100k_rate",

               "gdp_per_capita", "gdp_for_year"]

for name in columns_name:

    plt.figure(figsize=(15,6))

    plt.subplot(2,2,1)

    plt.hist(df[name])

    plt.title(name, fontdict=title_font)

        

    plt.subplot(2,2,2)

    plt.hist(np.log(df[name]+1))

    plt.title(name+ " (log transformation)", fontdict=title_font)

    plt.show()

    

        
columns_name= ["suicides_no", "population", "suicides/100k_rate",

               "gdp_per_capita", "gdp_for_year"]

for name in columns_name:

    q75_log, q25_log = np.percentile(np.log(df[name]), [75,25])

    caa_log= q75_log-q25_log

    q75, q25 = np.percentile(df[name], [75,25])

    caa= q75-q25

    comparison = pd.DataFrame(columns= ["threshold", "outliers {}".format(name), "outliers_log"])

    for threshold in np.arange(1,5,0.5):

        max_value_log = q75_log+ (caa_log*threshold)

        min_value_log = q25_log- (caa_log*threshold)

        max_value = q75+ (caa*threshold)

        min_value = q25- (caa*threshold)

        outliers_log = len((np.where((np.log(df[name]+1)>max_value_log) | 

                               (np.log(df[name]+1)<min_value_log))[0]))

        outliers = len((np.where((df[name]>max_value) | 

                     (df[name]<min_value))[0]))

        comparison = comparison.append({"threshold": threshold, "outliers {}".format(name): outliers,

                              "outliers_log": outliers_log}, ignore_index=True)

    display(comparison)   

    



df1 = pd.DataFrame(df.groupby("country").mean()["winsorize_gdp_per_capita"])

df2= pd.DataFrame(df.groupby("country").mean()["winsorize_suicides/100k_rate"])

df1["winsorize_suicides/100k_rate"]= df2["winsorize_suicides/100k_rate"]

df1.head()
def economy_convert(value):

    if value<5000:

        return "very low"

    elif value<10000:

        return "low"

    elif value<20000:

        return "medium"

    elif value<30000:

        return "high"

    else:

        return "very high"

df1["category"]= df1.winsorize_gdp_per_capita.apply(economy_convert)

df1
df1.groupby("category").mean()["winsorize_suicides/100k_rate"]
plt.figure(figsize=(10,6))

sns.barplot(df1["category"], df1["winsorize_suicides/100k_rate"], order= ["very low", "low", "medium", "high", "very high"])

plt.title("Economic Status And Suicide Rate", fontdict=title_font)

plt.xlabel("Economic Category", fontdict=label_font)

plt.ylabel("Suicide Rate", fontdict=label_font)

plt.xticks(color= "black")

plt.yticks(color= "black")

plt.show()

kategoriler = df1["category"].unique()

pd.options.display.float_format= "{:.6f}".format

karsilastirma = pd.DataFrame(columns= ["category_1", "category_2", "statistics", "p_value"])

for i in range(0,len(kategoriler)):

    for j in range(i+1, len(kategoriler)):

        ttest= stats.ttest_ind(df1[df1["category"]==kategoriler[i]]["winsorize_suicides/100k_rate"],

                               df1[df1["category"]==kategoriler[j]]["winsorize_suicides/100k_rate"])

        category_1 = kategoriler[i]

        category_2 = kategoriler[j]

        statistics = ttest[0]

        p_value = ttest[1]

        karsilastirma = karsilastirma.append({"category_1": category_1, "category_2": category_2,

                                              "statistics": statistics, "p_value": p_value}, ignore_index=True)

        

display(karsilastirma)   
df1 = pd.DataFrame(df.groupby("age").sum()["winsorize_suicides_no"]).reset_index()

df1.groupby("age").sum()["winsorize_suicides_no"]

df1
orderlist= ["5-14 years", "15-24 years", "25-34 years", "35-54 years", "55-74 years", "75+ years"]
plt.figure(figsize=(12,5))

sns.barplot(df["age"], df["winsorize_suicides_no"], order=orderlist)

plt.title("Suicide Numbers by Age Groups", fontdict=title_font)

plt.xlabel("Age Groups", fontdict=label_font)

plt.ylabel("Suicide Rates",  fontdict=label_font)

plt.xticks(color= "black")

plt.yticks(color= "black")

plt.show()  
yaslar = df1["age"].unique()

karsilastirma = pd.DataFrame(columns= ["group_1", "group_2", "statistics", "p_value"])

pd.options.display.float_format= "{:.6f}".format

for i in range(0,len(yaslar)):

    for j in range(i+1,len(yaslar)):

        ttest = stats.ttest_ind(df[df["age"]==yaslar[i]]["winsorize_suicides_no"],

                                df[df["age"]==yaslar[j]]["winsorize_suicides_no"])

        group_1 = yaslar[i]

        group_2 = yaslar[j]

        statistics = ttest[0]

        p_value = ttest[1]

        karsilastirma = karsilastirma.append({"group_1": group_1, "group_2": group_2,

                                              "statistics": statistics, "p_value": p_value}, ignore_index=True)

display(karsilastirma)
df1= df.groupby(["sex", "age"]).mean()["winsorize_suicides_no"].reset_index()





plt.figure(figsize=(20,5))

sns.barplot(data=df1, x= df1["age"], y=df1["winsorize_suicides_no"], hue=df1["sex"], order=orderlist)

plt.title("Suicide Rates By Gender", fontdict=title_font)

plt.xlabel("Age Groups", fontdict=label_font)

plt.ylabel("Suicide Rates", fontdict=label_font)



plt.show()  

df1 = pd.DataFrame()

df1["year"]= df["year"].astype("object")

df1["sex"]= df["sex"]

df1["winsorize_suicides/100k_rate"]= df["winsorize_suicides/100k_rate"]

df1.head()
plt.figure(figsize=(18,7))

sns.barplot(data=df1, x="year", y="winsorize_suicides/100k_rate", hue="sex")

plt.title("Suicide Rates by Years", fontdict=title_font)

plt.xlabel("Year", fontdict=label_font)

plt.ylabel("Suicide Rates", fontdict=label_font)

plt.xticks(color= "black")

plt.yticks(color= "black")

plt.show() 
yıllar = df["year"].unique()

karsilastirma = pd.DataFrame(columns= ["grup_1", "grup_2", "istatistik", "p_degeri"])

pd.options.display.float_format= "{:.6f}".format

for i in range(0,len(yıllar)):

    for j in range(i+1,len(yıllar)):

        ttest= stats.ttest_ind(df[df["year"]==yıllar[i]]["winsorize_suicides/100k_rate"],

                               df[df["year"]==yıllar[j]]["winsorize_suicides/100k_rate"])

        grup_1= yıllar[i]

        grup_2= yıllar[j]

        istatistik= ttest[0]

        p_degeri= ttest[1]

        karsilastirma= karsilastirma.append({"grup_1": grup_1, "grup_2": grup_2,

                                            "istatistik": istatistik, "p_degeri": p_degeri}, ignore_index=True)

display(karsilastirma[karsilastirma["p_degeri"]<0.005])
df.head()
df.dropna()

df1 = pd.DataFrame(df.groupby("country").mean()["HDI_for_year"])

df1["winsorize_suicides/100k_rate"]= df.groupby("country").mean()["winsorize_suicides/100k_rate"]

df1 = df1.dropna()

df1.head()
plt.figure(figsize=(12,6))

plt.scatter(df1["HDI_for_year"], df1["winsorize_suicides/100k_rate"])

plt.title("HDI And Suicide Rates", fontdict=title_font)

plt.xlabel("HDI", fontdict=label_font)

plt.ylabel("Suicide Rates", fontdict=label_font)

plt.show()  
df1.corr() 
korelasyon_matrisi_df1= df1.corr()

plt.figure(figsize=(14,6))

sns.heatmap(korelasyon_matrisi_df1, square=True, annot=True, linewidth=.5, vmin=0, vmax=1, cmap="Greens")

plt.title("Suicide Rate And HDI Correlation Matrix", fontdict=title_font)

plt.show()
df = df.dropna()

df2 = df[(df["country"]=="Brazil") | (df["country"]=="Mexico") |(df["country"]=="Turkey")] #Developing Countries

df2_ = pd.DataFrame(df2.groupby([df2["country"], df2["year"]]).mean()["winsorize_suicides/100k_rate"]).reset_index()

df3 = df[(df["country"]=="Argentina") | (df["country"]=="Chile") | (df["country"]=="Ecuador")] #South America

df3_ = pd.DataFrame(df3.groupby([df3["country"], df3["year"]]).mean()["winsorize_suicides/100k_rate"]).reset_index()

df4 = df[(df["country"]=="Belgium") | (df["country"]=="France") | (df["country"]=="Germany")] #Strong Economic Europe

df4_ = pd.DataFrame(df4.groupby([df4["country"], df4["year"]]).mean()["winsorize_suicides/100k_rate"]).reset_index()

df5 = df[(df["country"]=="Norway") | (df["country"]=="Finland") | (df["country"]=="Denmark")] #North Europe

df5_ = pd.DataFrame(df5.groupby([df5["country"], df5["year"]]).mean()["winsorize_suicides/100k_rate"]).reset_index()

df6 = df[(df["country"]=="Serbia") | (df["country"]=="Ukraine") |(df["country"]=="Bulgaria")] #East Europe

df6_ = pd.DataFrame(df6.groupby([df6["country"], df6["year"]]).mean()["winsorize_suicides/100k_rate"]).reset_index()

df7 = df[(df["country"]=="United States") | (df["country"]=="United Kingdom") |(df["country"]=="Japan")] #Countries with advanced technology

df7_ = pd.DataFrame(df7.groupby([df7["country"], df7["year"]]).mean()["winsorize_suicides/100k_rate"]).reset_index()



plt.figure(figsize=(20,6))

fig = px.line(df2_, x="year", y="winsorize_suicides/100k_rate", color="country")

fig.show()



plt.figure(figsize=(20,6))

fig= px.line(df3_, x="year", y="winsorize_suicides/100k_rate", color="country")

fig.show()



plt.figure(figsize=(20,6))

fig= px.line(df4_, x="year", y="winsorize_suicides/100k_rate", color="country")

fig.show()



plt.figure(figsize=(20,6))

fig= px.line(df5_, x="year", y="winsorize_suicides/100k_rate", color="country")

fig.show()



plt.figure(figsize=(20,6))

fig= px.line(df6_, x="year", y="winsorize_suicides/100k_rate", color="country")

fig.show()



plt.figure(figsize=(20,6))

fig= px.line(df7_, x="year", y="winsorize_suicides/100k_rate", color="country")

fig.show()





columns_name= ["suicides_no", "population", "suicides/100k_rate",

               "gdp_per_capita", "gdp_for_year"]

for name in columns_name:

    plt.figure(figsize=(15,6))

    plt.subplot(2,2,1)

    plt.hist(df[name])

    plt.title(name, fontdict=title_font)

        

    plt.subplot(2,2,2)

    plt.hist(np.log(df[name]+1))

    plt.title(name+ " (log transformation)", fontdict=title_font)

    plt.show()

    
columns_name = ["suicides_no", "suicides/100k_rate", "population", "gdp_per_capita", "gdp_for_year"]

pd.options.display.float_format = "{:.5f}".format



distribution_test = pd.DataFrame(columns= ["attribute", "jarque_bera_stats", "jarque_bera_p_value",

                                           "normaltest_stats", "normaltest_p_value"])

for names in columns_name:

    jb_stats= jarque_bera(np.log(df[names]+1))

    normal_stats= normaltest(np.log(df[names]+1))

    distribution_test= distribution_test.append({"attribute": names, 

                                                 "jarque_bera_stats": jb_stats[0], "jarque_bera_p_value": jb_stats[1],

                                                  "normaltest_stats": normal_stats[0], "normaltest_p_value": normal_stats[1]},

                                                ignore_index=True)

display(distribution_test)



#normalization:

df["norm_winsorize_suicides_no"]= normalize(np.array(df["winsorize_suicides_no"]).reshape(-1,1)).reshape(-1,1)

df["norm_winsorize_suicides/100k_rate"]= normalize(np.array(df["winsorize_suicides/100k_rate"]).reshape(-1,1)).reshape(-1,1)

df["norm_winsorize_gdp_per_capita"]= normalize(np.array(df["winsorize_gdp_per_capita"]).reshape(-1,1)).reshape(-1,1)

df["norm_winsorize_gdp_for_year"]= normalize(np.array(df["winsorize_gdp_for_year"]).reshape(-1,1)).reshape(-1,1)

df["norm_winsorize_population"]= normalize(np.array(df["winsorize_population"]).reshape(-1,1)).reshape(-1,1)



norm_features= ["winsorize_suicides_no", "norm_winsorize_suicides_no",

                "winsorize_suicides/100k_rate", "norm_winsorize_suicides/100k_rate",

                "winsorize_gdp_per_capita", "norm_winsorize_gdp_per_capita",

                "winsorize_gdp_for_year", "norm_winsorize_gdp_for_year",

                "winsorize_population", "norm_winsorize_population"]

print("---Minimum Values---\n ")

print(df[norm_features].min())

print("---Maximum Values---\n ")

print(df[norm_features].max())
# standardization

df["scale_suicides_no"]= scale(df["winsorize_suicides_no"])

df["scale_suicides/100k_rate"]= scale(df["winsorize_suicides/100k_rate"])

df["scale_gdp_per_capita"]= scale(df["winsorize_gdp_per_capita"])

df["scale_gdp_for_year"]= scale(df["winsorize_gdp_for_year"])

df["scale_population"]= scale(df["winsorize_population"])



scale_features= ["winsorize_suicides_no", "scale_suicides_no", 

                 "winsorize_suicides/100k_rate", "scale_suicides/100k_rate", 

                 "winsorize_gdp_per_capita", "scale_gdp_per_capita", 

                 "winsorize_gdp_for_year", "scale_gdp_for_year", 

                 "winsorize_population", "scale_population"]

print("---Standard Deviation--- \n")

print(df[scale_features].std())

print("---Means--- \n")

print(df[scale_features].mean())
