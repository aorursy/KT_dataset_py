# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

data.shape
data.info()
dataEdit = data.drop(columns = "HDI for year")

dataEdit.rename(columns={"country":"Country",'suicides/100k pop':'suicid_ratio_100K','gdp_for_year ($)':'gdp_for_Year','gdp_per_capita ($)':'gdp_per_capita'},inplace=True)

dataEdit.loc[dataEdit["generation"]=="Generation Z"]

dataEdit.head()
countryData = dataEdit.Country.value_counts().rename_axis("Country").reset_index(name='Number Of Data')

countryData.head()
countryData.describe()

dataEdit.loc[dataEdit["Country"]=="Macau"]
dataEdit.loc[dataEdit["Country"]=="Austria"].head()

countryYear = dataEdit.groupby(["Country","year"])

countryYearCount = countryYear.size().reset_index(name='Count')

countryYearCount = countryYearCount.loc[countryYearCount["Count"]==12]

countryYearCount.head()
countryYearCount.loc[countryYearCount["year"]==2016]
countryYearCountValue = countryYearCount.Country.value_counts().rename_axis('Country').reset_index(name='NumberOfYears')

FreqCountryCount = countryYearCountValue.NumberOfYears.value_counts().rename_axis('Number Of Years').reset_index(name='Count').sort_values(by=['Number Of Years'],ascending=False)

FreqCountryCount.head()


fig, ax = plt.subplots(figsize=(10,26))

bar = sns.scatterplot(x="year", y="Country", data=countryYearCount)



countries31 = countryYearCountValue.loc[countryYearCountValue["NumberOfYears"]==31]

countries31.head()
def OverallRatio(factor,order):

    result={}

    ratioDataPoints={}

    ratioPopulation = {}

    totalDataPoints = dataEdit.size

    totalPopulation = dataEdit["population"].sum()

    for i in order:

        ratioDataPoints[i] = dataEdit.loc[dataEdit[factor]==i].size/totalDataPoints

        ratioPopulation[i] = dataEdit.loc[dataEdit[factor]==i]["population"].sum()/totalPopulation

    result["ratioDataPoints"]=ratioDataPoints

    result["ratioPopulation"]=ratioPopulation

    return result



def toDataFrame(data,index,columnName):    

    dataFrame = pd.Series(data, name= columnName)

    dataFrame.index.names=[index]

    dataFrame=dataFrame.reset_index(name=columnName)

    return dataFrame



ratioDataPoints = toDataFrame(data=OverallRatio(factor="sex",order=["male","female"])["ratioDataPoints"],index="Sex",columnName="Ratio")

ratioPopulation = toDataFrame(data=OverallRatio(factor="sex",order=["male","female"])["ratioPopulation"],index="Sex",columnName="Ratio")

ratioDataPoints,ratioPopulation

f, ax = plt.subplots(figsize=(15, 10),ncols=2)

sns.set_color_codes("pastel")

plot = sns.barplot(x="Sex", y="Ratio", data=ratioDataPoints,label="Total", color="b" ,ci=None, order=["male","female"],ax=ax[0]).set_title("Data Points")

plot = sns.barplot(x="Sex", y="Ratio", data=ratioPopulation,label="Total", color="b" ,ci=None, order=["male","female"],ax=ax[1]).set_title("Population")
def averageSuicideRatio(factor,order):

    ratio={}

    for i in order:

        ratio[i] = dataEdit.loc[dataEdit[factor]==i]["suicid_ratio_100K"].mean()

    return ratio



average_Suicide_Ratio = toDataFrame(data=averageSuicideRatio(factor="sex",order=["male","female"]),index="Sex",columnName="Average Suicide Ratio per 100k")

average_Suicide_Ratio



def TotalSuicides(groupby,seperate,suicides_no):

    result = []

    groupbyLength = len(groupby)

    if seperate == True:

        for i in FreqCountryCount["Number Of Years"]:

            countries = countryYearCountValue.loc[countryYearCountValue["NumberOfYears"]==i]

            CountryAnalysis=pd.DataFrame() 

            for country in countries["Country"]:

                countrydf = dataEdit.loc[dataEdit["Country"]==country]

                CountryAnalysis = CountryAnalysis.append(countrydf,ignore_index=True)



            #drop year of 2016

            CountryAnalysis = CountryAnalysis.drop(CountryAnalysis[CountryAnalysis.year==2016].index)

            #Analysis based on groupby

            if suicides_no==True:

                CountryAnalysisTotalSuicides = CountryAnalysis.groupby(groupby).agg({"suicides_no":"sum"})

                CountryAnalysisTotalSuicides.reset_index(level=list(range(groupbyLength)), inplace=True)

                result.append([CountryAnalysisTotalSuicides,i])

            else:

                CountryAnalysisTotalSuicides = CountryAnalysis.groupby(groupby).agg({"suicid_ratio_100K":"mean"})

                CountryAnalysisTotalSuicides.reset_index(level=list(range(groupbyLength)), inplace=True)

                result.append([CountryAnalysisTotalSuicides,i])

        return result

    else:

        if suicides_no==True:

            CountryAnalysisTotalSuicides = dataEdit.groupby(groupby).agg({"suicides_no":"sum","gdp_per_capita":"mean","population":"sum"})

            CountryAnalysisTotalSuicides.reset_index(level=list(range(groupbyLength)), inplace=True)

            return CountryAnalysisTotalSuicides

        else:

            CountryAnalysisTotalSuicides = dataEdit.groupby(groupby).agg({"suicid_ratio_100K":"mean","gdp_per_capita":"mean"})

            CountryAnalysisTotalSuicides.reset_index(level=list(range(groupbyLength)), inplace=True)

            return CountryAnalysisTotalSuicides





def TotalSuicidesSex(all_or_numbers):

    if all_or_numbers == True:

        for i in TotalSuicides(["Country","sex"],seperate = True,suicides_no=True):

            sns.set(style="whitegrid")

            plot = sns.catplot(x="Country", y="suicides_no", data=i[0] ,hue="sex", kind="bar", palette="muted",height=15, aspect=2, ci=None)

            plt.title("Total Suicides From 1985 to 2015 -- {} Years worth of Data ".format(i[1]))

            plot.set_axis_labels("", "Number of Suicidies")

        return plot

    else:

        for i in TotalSuicides(["Country","sex"],seperate = True,suicides_no=True):

            if i[1] in all_or_numbers:

                sns.set(style="whitegrid")

                plot = sns.catplot(x="Country", y="suicides_no", data=i[0] ,hue="sex", kind="bar", palette="muted",height=15, aspect=2, ci=None)

                plt.title("Total Suicides From 1985 to 2015 -- {} Years worth of Data ".format(i[1]))

                plot.set_axis_labels("", "Number of Suicidies")

        return plot    



TotalSuicidesSex(all_or_numbers=[31,30])
def Ratio(data,factors,name,numerator,denominator,suicides_no):

    Ratio = data

    if suicides_no==True:

        c = Ratio.groupby(factors)["suicides_no"].sum().reset_index(name = denominator)

    else:

        c = Ratio.groupby(factors)["suicid_ratio_100K"].mean().reset_index(name = denominator)

    Ratio = pd.merge(Ratio, c, on=factors, how='outer')

    Ratio[name] = Ratio[numerator]/Ratio[denominator]

    Ratio[name] = Ratio[name].replace('nan', np.nan).fillna(0)

    return Ratio



MenVsWomen = Ratio(TotalSuicides(["Country","sex"],seperate=False,suicides_no=True),factors='Country',name="Ratio",numerator="suicides_no",denominator="Total" ,suicides_no=True)

MenVsWomenNoSuicide =  MenVsWomen.loc[MenVsWomen["Total"]==0]

MenVsWomen = MenVsWomen.loc[MenVsWomen["Total"]>0]

onlyMales = MenVsWomen.loc[MenVsWomen["sex"]=="male"]

boxplot = sns.boxplot(x=onlyMales["Ratio"])

maximum = onlyMales.loc[onlyMales['Ratio'].idxmax()]["Country"]

minimum = onlyMales.loc[onlyMales['Ratio'].idxmin()]["Country"]

boxplot.text(0.97, 0.1, maximum, horizontalalignment='left', size='medium', color='black', weight='semibold')

boxplot.text(0.55, 0.1, minimum, horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.title("Ratio of Suicides by Males")

description = onlyMales.describe()

description=description["Ratio"].reset_index(name='Ratio')

description.rename(columns={'index':'Description'},inplace=True)

description

MenVsWomenNoSuicide
age_order=["5-14 years","15-24 years","25-34 years","35-54 years","55-74 years","75+ years"]

ratioDataPoints = toDataFrame(data=OverallRatio(factor="age",order=age_order)["ratioDataPoints"],index="Age Groups",columnName="Ratio")

ratioPopulation = toDataFrame(data=OverallRatio(factor="age",order=age_order)["ratioPopulation"],index="Age Groups",columnName="Ratio")

ratioDataPoints,ratioPopulation

f, ax = plt.subplots(figsize=(25, 10),ncols=2)

sns.set_color_codes("pastel")

plot = sns.barplot(x="Age Groups", y="Ratio", data=ratioDataPoints,label="Total", color="b" ,ci=None, order=age_order,ax=ax[0]).set_title("Data Points")

plot = sns.barplot(x="Age Groups", y="Ratio", data=ratioPopulation,label="Total", color="b" ,ci=None, order=age_order,ax=ax[1]).set_title("Population")
age_order=["5-14 years","15-24 years","25-34 years","35-54 years","55-74 years","75+ years"]

def averageSuicideRatio(factor,order):

    ratio={}

    for i in order:

        ratio[i] = dataEdit.loc[dataEdit[factor]==i]["suicid_ratio_100K"].mean()

    return ratio



average_Suicide_Ratio = toDataFrame(data=averageSuicideRatio(factor="age",order=age_order),index="Age",columnName="Average Suicide Ratio per 100k")

average_Suicide_Ratio

#groupSexRatio(groupName = "age",order=age_order,suicides_no=True)
average_Suicide_Ratio = toDataFrame(data=averageSuicideRatio(factor="age",order=age_order),index="Age Groups",columnName="Average Suicide Ratio per 100k")

f, ax = plt.subplots(figsize=(15, 10))

plot = sns.barplot(x="Age Groups", y="Average Suicide Ratio per 100k", data=average_Suicide_Ratio,

                color="b",ci=None, order=age_order)
def RatioSuicidesPerCounty(all_or_numbers,factor,order):

    if all_or_numbers == True:

        for i in TotalSuicides(["Country",factor],seperate = True,suicides_no=False):

            sns.set(style="whitegrid")

            plot = sns.catplot(x="Country", y="suicid_ratio_100K", data=i[0] ,hue=factor, kind="bar", palette="muted",height=15, aspect=2,hue_order=order)

            plt.title("Total Suicides From 1985 to 2015 -- {} Years worth of Data ".format(i[1]))

            plot.set_axis_labels("", "Number of Suicidies per 100k")

        return plot

    else:

        for i in TotalSuicides(["Country",factor],seperate = True,suicides_no=False):

            if i[1] in all_or_numbers:

                sns.set(style="whitegrid")

                plot = sns.catplot(x="Country", y="suicid_ratio_100K", data=i[0] ,hue=factor, kind="bar", palette="muted",height=15, aspect=2,hue_order=order)

                plt.title("Total Suicides From 1985 to 2015 -- {} Years worth of Data ".format(i[1]))

                plot.set_axis_labels("", "Number of Suicidies per 100k")

        return plot

 

RatioSuicidesPerCounty(all_or_numbers=[30,31],factor="age",order=age_order)
def groupSexRatio (groupName,order,suicides_no):

    if suicides_no==False:

        groupSexRatio = Ratio(TotalSuicides([groupName,"sex"],seperate = False,suicides_no=False),

                          factors=[groupName],name="Ratio",numerator="suicid_ratio_100K",denominator="Total",suicides_no=False)

    else:

        groupSexRatio = Ratio(TotalSuicides([groupName,"sex"],seperate = False,suicides_no=True),

                          factors=[groupName],name="Ratio",numerator="suicides_no",denominator="Total",suicides_no=True)

    



    sns.set(style="whitegrid")







    # Plot the Males

    if suicides_no==False:

        sns.set_color_codes("muted")

        plot = sns.catplot(x=groupName, y="suicid_ratio_100K", data=groupSexRatio,

                    color="b",order=order,hue="sex", kind="bar", palette="muted",height=15, aspect=2)

        plot.set_axis_labels("", "Number of Suicidies")

        return plot

    else:

        groupSexRatio = groupSexRatio.loc[groupSexRatio["sex"]=="male"]

        f, ax = plt.subplots(figsize=(15, 10))

        # Plot the total 

        sns.set_color_codes("pastel")

        plot = sns.barplot(x=groupName, y="Total", data=groupSexRatio,

        label="Total", color="b" ,ci=None, order=order)

        sns.set_color_codes("muted")

        plot = sns.barplot(x=groupName, y="suicides_no", data=groupSexRatio,label="Male",

                            color="b",ci=None, order=order)

        ax.legend(ncol=2, loc="upper right", frameon=True)

        ax.set( ylabel="Total Number of Suicids",

               xlabel=groupName+ "Groups")

        return plot
age_order=["5-14 years","15-24 years","25-34 years","35-54 years","55-74 years","75+ years"]

groupSexRatio(groupName = "age",order=age_order,suicides_no=True)
def TableSexGroupRatio(groupName):

    GroupRatio = Ratio(TotalSuicides(["Country",groupName,"sex"],seperate = False,suicides_no=True),

               factors='country',name="Ratio "+ groupName +" Group",numerator="suicides_no",denominator="Total "+groupName+ " Group",suicides_no=True )

    c = GroupRatio.groupby(['country',groupName])["suicides_no"].sum().reset_index(name = "Total Sex of Age Group")

    GroupRatio = pd.merge(GroupRatio, c, on=['country',groupName], how='outer')

    GroupRatio["Ratio Sex of "+groupName+" Group"] = GroupRatio["Total Sex of Age Group"]/GroupRatio["Total "+groupName+ " Group"]

    GroupRatio["Ratio Sex of "+groupName+" Group"] = GroupRatio["Ratio Sex of "+groupName+" Group"].replace('nan', np.nan).fillna(0)



    return GroupRatio
def GroupCountrySexRatio (Data,groupName,hue_order_total,hue_order_group,labels):

    data = Data.loc[Data["Total " +groupName+" Group"]>0]

    for i in [31]:

        countries = countryYearCountValue.loc[countryYearCountValue["NumberOfYears"]==i]

        CountryAnalysis=pd.DataFrame() 

        for country in countries["Country"]:

            countrydf = data.loc[data["Country"]==country]

            CountryAnalysis = CountryAnalysis.append(countrydf,ignore_index=True)

        sns.set(style="whitegrid")

        f,ax = plt.subplots(figsize=(20, 30))

        # Plot the total 

        sns.catplot(x="Ratio Sex of " +groupName+ " Group", y="Country", data=CountryAnalysis,hue=groupName,

                     kind="bar", palette="dark",ax=ax, ci=None ,hue_order=hue_order_total)

        # Plot the Males

        sns.catplot(x="Ratio "+groupName+" Group", y="Country", data=CountryAnalysis,hue=groupName,

                    kind="bar", palette="pastel",ax=ax, ci=None,hue_order=hue_order_group)

        

        ax.legend(ncol=2, loc="upper right", frameon=True,labels=labels)

        ax.set( ylabel="Countries with {} Years of Data ".format(i),xlabel="Ratio in terms of "+groupName+" group")

        plt.close(2)

        plt.close(3)

        plt.show()

    
def CountGroupRatio(Data,groupName,order):

    result={}

    table = Data

    countries = table.Country.unique()

    groupdf=pd.DataFrame()

    for country in countries:

        newRow = pd.DataFrame()    

        ratioMax=table.loc[table["Country"]==country]["suicid_ratio_100K"].max()

        row =table.loc[(table["Country"]==country) & (table["suicid_ratio_100K"] == ratioMax)]

        newRow["Country"]=row["Country"]

        newRow[groupName]=row[groupName]

        newRow["sex"]=row["sex"]

        newRow["suicid_ratio_100K"]=row["suicid_ratio_100K"]

        groupdf=groupdf.append(newRow,ignore_index = True)

    groupName_count = groupdf.groupby([groupName,"sex"]).size().reset_index(name='Count')

    groupName_count["Ratio"] = groupName_count['Count']/groupName_count['Count'].sum()



    sns.set(style="whitegrid")

    f,ax = plt.subplots(figsize=(10, 7))

    # Plot the total 

    plot = sns.catplot(x=groupName, y="Count", data=groupName_count,hue="sex",

                 kind="bar", palette="pastel",ax=ax, ci=None, order=order)

    plt.close(2)

    return groupName_count



data =Ratio(TotalSuicides(["Country","age","sex"],seperate = False,suicides_no=False),

               factors='Country',name="Ratio "+ "age" +" Group",numerator="suicid_ratio_100K",denominator="Total "+"age"+ " Group",suicides_no=False )

order = ["5-14 years","15-24 years","25-34 years","35-54 years","55-74 years","75+ years"]

CountGroupRatio(Data=data,groupName="age",order=order)



gen_order=["G.I. Generation","Silent","Boomers","Generation X","Millenials","Generation Z"]

ratioDataPoints = toDataFrame(data=OverallRatio(factor="generation",order=gen_order)["ratioDataPoints"],index="Generation Groups",columnName="Ratio")

ratioPopulation = toDataFrame(data=OverallRatio(factor="generation",order=gen_order)["ratioPopulation"],index="Generation Groups",columnName="Ratio")

f, ax = plt.subplots(figsize=(25, 10),ncols=2)

sns.set_color_codes("pastel")

plot = sns.barplot(x="Generation Groups", y="Ratio", data=ratioDataPoints,label="Total", color="b" ,ci=None, order=gen_order,ax=ax[0]).set_title("Data Points")

plot = sns.barplot(x="Generation Groups", y="Ratio", data=ratioPopulation,label="Total", color="b" ,ci=None, order=gen_order,ax=ax[1]).set_title("Population")

def AgeGeneration ():

    age_order=["5-14 years","15-24 years","25-34 years","35-54 years","55-74 years","75+ years"]

    gen_order=["G.I. Generation","Silent","Boomers","Generation X","Millenials","Generation Z"]

    TableAgeGenerationRatio = Ratio(TotalSuicides(["generation","age"],seperate = False,suicides_no=True),factors="generation",

                                    name="Ratio age given generation",numerator="suicides_no",denominator="Total Suicides in the Generation",suicides_no=True )



    tableGenAge = TableAgeGenerationRatio.pivot(index='generation', columns='age', values='suicides_no')

    tableGenAge.fillna(0)

    tableGenAge.reindex(gen_order)

    tableGenAge1 = TableAgeGenerationRatio.pivot(index='age', columns='generation', values='suicides_no')

    tableGenAge1.reindex(age_order)

    tableGenAge1.fillna(0)

    palette = iter(sns.husl_palette(2*len(age_order)))

    sns.set(style="whitegrid")

    f, ax = plt.subplots(figsize=(15, 10),ncols=2)

    # Plot the other age groups

    for age in reversed(age_order):

        filterTable=tableGenAge[age].reindex(gen_order).reset_index(name="suicides_no").fillna(0)

        filterTable["result"]=0    

        for gen in gen_order:

            filterTable1=tableGenAge1[gen].reindex(age_order).reset_index(name="suicides_no").fillna(0)

            filterTable1["culum"]=filterTable1["suicides_no"].cumsum()

            value = filterTable1.loc[filterTable1["age"]==age]["culum"].get_values()

            filterTable.loc[filterTable["generation"]==gen,["result"]]=value

        plot = sns.barplot(x="generation",y="result", data=filterTable,

                label=age, color=next(palette),ci=None, order=gen_order ,ax=ax[0])

        ax[0].legend(ncol=2, loc="upper right", frameon=True)

        ax[0].set( ylabel="Number of Suicides",

               xlabel="Generations")

    tableGenAge = TableAgeGenerationRatio.pivot(index='generation', columns='age', values='Ratio age given generation')

    tableGenAge.fillna(0)

    tableGenAge.reindex(gen_order)

    tableGenAge1 = TableAgeGenerationRatio.pivot(index='age', columns='generation', values='Ratio age given generation')

    tableGenAge1.reindex(age_order)

    tableGenAge1.fillna(0)

    for age in reversed(age_order):

        filterTable=tableGenAge[age].reindex(gen_order).reset_index(name="Ratio age given generation").fillna(0)

        filterTable["result"]=0    

        for gen in gen_order:

            filterTable1=tableGenAge1[gen].reindex(age_order).reset_index(name="Ratio age given generation").fillna(0)

            filterTable1["culum"]=filterTable1["Ratio age given generation"].cumsum()

            value = filterTable1.loc[filterTable1["age"]==age]["culum"].get_values()

            filterTable.loc[filterTable["generation"]==gen,["result"]]=value

        plot = sns.barplot(x="generation",y="result", data=filterTable,

                label=age, color=next(palette),ci=None, order=gen_order,ax=ax[1] )

        ax[1].legend(ncol=2, loc="upper right", frameon=True)

        ax[1].set( ylabel="Ratio of Suicides",

               xlabel="Generations")

    return plot
AgeGeneration()
groupSexRatio(groupName = "generation",order=["G.I. Generation","Silent","Boomers","Generation X","Millenials","Generation Z"],suicides_no=True)
TableAgeGenerationRatio = Ratio(TotalSuicides(["generation","age"],seperate = False,suicides_no=True),factors="generation",name="Ratio age given generation",

                                numerator="suicides_no",denominator="Total Suicides in the Generation",suicides_no=True )
average_Suicide_Ratio = toDataFrame(data=averageSuicideRatio(factor="generation",order=gen_order),index="Generation Groups",columnName="Average Suicide Ratio per 100k")

f, ax = plt.subplots(figsize=(15, 10))

plot = sns.barplot(x="Generation Groups", y="Average Suicide Ratio per 100k", data=average_Suicide_Ratio,color="b",ci=None, order=gen_order)
RatioSuicidesPerCounty(all_or_numbers=[30,31],factor="generation",order=gen_order)
order=["G.I. Generation","Silent","Boomers","Generation X","Millenials","Generation Z"]

data =Ratio(TotalSuicides(["Country","generation","sex"],seperate = False,suicides_no=False),

               factors='Country',name="Ratio "+ "generation" +" Group",numerator="suicid_ratio_100K",denominator="Total "+"generation"+ " Group",suicides_no=False )

CountGroupRatio(Data=data,groupName="generation",order=order)

dataPoints = dataEdit.groupby(["Country","year"]).size().reset_index(name="count")

dataPoints = dataPoints.groupby(["Country"]).size().reset_index(name="count")

dataPoints = dataPoints.loc[dataPoints["count"]>3]

import numpy as np

from scipy import stats



def pearsonr_ci(x,y,alpha=0.05):

    #    r : Pearson's correlation coefficient/ p : The corresponding p value/ lo, hi : The lower and upper bound of confidence intervals

    r, p = stats.pearsonr(x,y)

    r_z = np.arctanh(r)

    se = 1/np.sqrt(x.size-3)

    z = stats.norm.ppf(1-alpha/2)

    lo_z, hi_z = r_z-z*se, r_z+z*se

    lo, hi = np.tanh((lo_z, hi_z))

    return r, p, lo, hi


data =TotalSuicides(["Country","year"],seperate = False,suicides_no=True)

data["ratio"]=data["suicides_no"]/data["population"]*100000

f, ax = plt.subplots(figsize=(70, 70),nrows=10, ncols=10)

columns=("Country", "Correlation", "P-Value","Confidence Interval Lower","Confidence Interval Higher")

corrCountry =  pd.DataFrame(columns=columns)

for counter, country in enumerate (dataPoints.Country.unique()):

    x = int(counter/10)

    y= counter%10

    dataCountry = data.loc[data["Country"]==country]

    xdata = dataCountry["ratio"]

    ydata = dataCountry["gdp_per_capita"]

    corr = pearsonr_ci(xdata,ydata,alpha=0.05)

    row=[country]

    for i in corr:

        row.append(i)

    corrCountry.loc[counter]=row

    plot = sns.regplot(data=dataCountry.loc[dataCountry["Country"]==country],x="ratio",y="gdp_per_capita",ax=ax[x][y],color="g").set_title(country)





fig, ax = plt.subplots(figsize=(20,26))

g=sns.pointplot(x="Correlation", y="Country", data=corrCountry,join=False,ax=ax)

Y = corrCountry["Country"].values

X = corrCountry["Correlation"].values

corrCountry["low"] = corrCountry["Correlation"]-corrCountry["Confidence Interval Lower"]

corrCountry["high"] = corrCountry["Confidence Interval Higher"]-corrCountry["Correlation"]

L = corrCountry["low"].values

H = corrCountry["high"].values

error=[L,H]

g.errorbar(X, Y, xerr=error,fmt='o', capsize=2, elinewidth=1.1)

g.set_title("Correlation of GDP and ratio of Suicides per 100K with confidence Interval")

fig, ax = plt.subplots(figsize=(30,10))

bins = np.arange(-1,1,0.1)

x_axis = []

for i in range(0,len(bins)-1):

    point = [round(bins[i],1),round(bins[i+1],1)]

    x_axis.append(point)  

corrCountry['binned'] = pd.cut(corrCountry['Correlation'], bins)



plot = sns.countplot(x="binned",data=corrCountry)

plot.set_xticklabels(x_axis, rotation=30)

plot.set_xlabel("Correlation",labelpad = 30)

plot.set_ylabel("Count",labelpad = 30)

plot.set_title("Correlation of GDP per Capita and suicide Ratio per 100K",pad = 30)
correlationDescribe = corrCountry["Correlation"].describe()

correlationDescribe
dataHDI = pd.read_csv('../input/human-development-index-hdi/HDI.csv')

dataHDI1 = pd.read_csv('../input/human-development-index-hdi/HDI.csv')

dataHDICountry = dataHDI["Country"].unique()

dataEditCountry = dataEdit["Country"].unique()

countriesNotInHDI = []

for country in dataEditCountry:

    if country not in dataHDICountry:

        countriesNotInHDI.append(country)

print(countriesNotInHDI)


oldColumnName = dataHDI.columns

newColumnName= dataHDI1.columns.str.replace(" ","")

diffColumn = dict(zip(oldColumnName, newColumnName))

dataHDI1.rename(columns=diffColumn, inplace=True)

HDI2015 = dataHDI1.filter(regex="^[A-Za-z()%,-]+2015{1}|^[A-Za-z()%,-]+$")

HDI2015Column=HDI2015.columns

newDiffColumn={}

for col in HDI2015Column:

    newDiffColumn[col]=[k for k,v in diffColumn.items() if v == col][0]

HDI2015.rename(columns=newDiffColumn, inplace=True)

HDI2015=HDI2015.drop(columns=["Id",'Population Median age (years) 2015',"Total Population (millions) 2015","Population Urban 2015 %"])


dataEdit2015 = dataEdit.loc[dataEdit["year"]==2015]

dataEdit2015Country=dataEdit2015.groupby(["Country"]).agg({"suicides_no":"sum","population":"sum"}).reset_index()

dataEdit2015Country["ratio"]=dataEdit2015Country["suicides_no"]/dataEdit2015Country["population"]*100000

AllDataCountry = pd.merge(dataEdit2015Country, HDI2015, on='Country')

AllDataCountry=AllDataCountry.dropna()

def correlationToRatio(data,matrix):

    columns=("ColName", "Correlation", "P-Value","Confidence Interval Lower","Confidence Interval Higher")

    corrData = pd.DataFrame(columns=columns)

    indexRatio = data.columns.tolist().index("ratio")

    for i in range(indexRatio,len(data.columns.tolist())):

        xData = data["ratio"]

        colName = data.columns.tolist()[i]

        yData=data[colName]

        corr=pearsonr_ci(x=xData,y=yData,alpha=0.05)

        row=[colName]

        for j in corr:

            row.append(j)

        corrData.loc[i]=row

    a = corrData[corrData["Correlation"].isnull()]

    corrData=corrData.dropna()

    corrData=corrData[corrData.ColName != 'ratio']

    numberCol=corrData.shape[0]

    if matrix==True:

        f, ax = plt.subplots(figsize=(70, 70),nrows=int(numberCol/5)+1, ncols=5)

        for counter,colName in enumerate (corrData.ColName.unique()):

            x = int(counter/5)

            y= counter%5

            plot = sns.regplot(data=data,x="ratio",y=colName,ax=ax[x][y],color="g").set_title(colName,pad=20)

    fig, ax1 = plt.subplots(figsize=(20,15))

    g=sns.pointplot(x="Correlation", y="ColName", data=corrData,join=False,ax=ax1)

    Y = corrData["ColName"].values

    X = corrData["Correlation"].values

    corrData["low"] = corrData["Correlation"]-corrData["Confidence Interval Lower"]

    corrData["high"] = corrData["Confidence Interval Higher"]-corrData["Correlation"]

    L = corrData["low"].values

    H = corrData["high"].values

    error=[L,H]

    g.errorbar(X, Y, xerr=error,fmt='o', capsize=2, elinewidth=1.1)

    g.set_title("Correlation of Social Economic Factors and ratio of Suicides per 100K with confidence Interval")

    plt.xlim(-1, +1)



    return 

        
correlationToRatio(data=AllDataCountry,matrix=True)
dataEdit2015 = dataEdit.loc[dataEdit["year"]==2015]

dataEdit2015Sex=dataEdit2015.groupby(["Country","sex"]).agg({"suicides_no":"sum"}).reset_index()

c=dataEdit2015Sex.groupby(["Country"]).agg({"suicides_no":"sum"}).reset_index()

dataEdit2015Sex = pd.merge(dataEdit2015Sex, c, on="Country", how='outer')

dataEdit2015Sex.rename(columns={"suicides_no_x":"suicides","suicides_no_y":"Total Suicides"},inplace=True)

dataEdit2015Sex = dataEdit2015Sex.loc[dataEdit2015Sex["Total Suicides"]>0]

dataEdit2015Sex["ratio"]=dataEdit2015Sex["suicides"]/dataEdit2015Sex["Total Suicides"]

dataEdit2015Sex=dataEdit2015Sex[dataEdit2015Sex.sex != 'female']

AllDataCountrySex = pd.merge(dataEdit2015Sex, HDI2015, on='Country')



correlationToRatio(data=AllDataCountrySex.dropna(),matrix=False)
dataEdit2015 = dataEdit.loc[dataEdit["year"]==2015]

dataEdit2015Sex=dataEdit2015.groupby(["Country","sex"]).agg({"suicides_no":"sum","population":"sum"}).reset_index()

c=dataEdit2015Sex.groupby(["Country"]).agg({"suicides_no":"sum","population":"sum"}).reset_index()

dataEdit2015Sex = pd.merge(dataEdit2015Sex, c, on="Country", how='outer')

dataEdit2015Sex=dataEdit2015Sex.drop(columns=["population_x","suicides_no_y"])

dataEdit2015Sex.rename(columns={"suicides_no_x":"suicides","population_y":"Total Population"},inplace=True)

dataEdit2015Sex = dataEdit2015Sex.loc[dataEdit2015Sex["Total Population"]>0]

dataEdit2015Sex["ratio"]=dataEdit2015Sex["suicides"]/dataEdit2015Sex["Total Population"]*100000

dataEdit2015Sex = dataEdit2015Sex.dropna()



def correlationToRatioGroup(data,dataHDI,orders,group):



    columns=("ColName", "Correlation", "P-Value","Confidence Interval Lower","Confidence Interval Higher","GroupFactor")

    corrDataTotal = pd.DataFrame(columns=columns)

    for count, order in enumerate (orders):

        corrData = pd.DataFrame(columns=columns)

        groupData=data.loc[data[group] == order]

        AllDataCountryGroup = pd.merge(groupData, dataHDI, on='Country').dropna()       

        indexRatio = data.columns.tolist().index("ratio")

        for i in range(indexRatio+1,len(AllDataCountryGroup.columns.tolist())):

            xData = AllDataCountryGroup["ratio"]

            colName = AllDataCountryGroup.columns.tolist()[i]

            yData=AllDataCountryGroup[colName]

            corr=pearsonr_ci(x=xData,y=yData,alpha=0.05)

            row=[colName]

            for j in corr:

                row.append(j)

            row.append(order)

            corrData.loc[i]=row

        corrDataTotal=corrDataTotal.append(corrData)

    corrDataTotal=corrDataTotal.dropna()

    corrDataTotal=corrDataTotal.loc[corrDataTotal.ColName != 'ratio']

    fig, ax1 = plt.subplots(figsize=(20,len(AllDataCountryGroup.columns.tolist())*0.3+2))

    g=sns.pointplot(x="Correlation", y="ColName", data=corrDataTotal,join=False,ax=ax1,hue="GroupFactor")

    color = ["blue","orange","green","red","grey","brown"]

    for count, order in enumerate (orders):

        filteredData = corrDataTotal.loc[corrDataTotal["GroupFactor"]==order]    

        Y = filteredData["ColName"].values

        X = filteredData["Correlation"].values

        filteredData["low"] = filteredData["Correlation"]-filteredData["Confidence Interval Lower"]

        filteredData["high"] = filteredData["Confidence Interval Higher"]-filteredData["Correlation"]

        L = filteredData["low"].values

        H = filteredData["high"].values

        error=[L,H]

        g.errorbar(X, Y, xerr=error,fmt='o', capsize=2, elinewidth=1.1,color=color[count])

    g.set_title("Correlation of Social Economic Factors and ratio of Suicides per 100K with confidence Interval for different sexes")  

    plt.xlim(-1, +1)



   

    return 

correlationToRatioGroup(data=dataEdit2015Sex,dataHDI=HDI2015,orders=["female","male"],group="sex")
HDISex = dataHDI[["Country",'Total fertility rate (birth per woman) 2000/2007']]

dataEdit2000_2007 = dataEdit.loc[dataEdit["year"].isin(["2000","2001","2002","2003","2004","2005","2006","2007"])]

dataEdit2000_2007Sex=dataEdit2000_2007.groupby(["Country","year","sex"]).agg({"suicides_no":"sum","population":"sum"}).reset_index()

dataEdit2000_2007Sex

dataEdit2000_2007Sex["ratio"]=dataEdit2000_2007Sex["suicides_no"]/dataEdit2000_2007Sex["population"]*100000

dataEdit2000_2007Sex=dataEdit2000_2007Sex.groupby(["Country","sex"]).agg({"ratio":"mean"}).reset_index()
correlationToRatioGroup(data=dataEdit2000_2007Sex,dataHDI=HDISex,orders=["female","male"],group="sex")
dataEdit2015 = dataEdit.loc[dataEdit["year"]==2015]

dataEdit2015Age=dataEdit2015.groupby(["Country","age"]).agg({"suicides_no":"sum","population":"sum"}).reset_index()

c=dataEdit2015Age.groupby(["Country"]).agg({"suicides_no":"sum","population":"sum"}).reset_index()

dataEdit2015Age = pd.merge(dataEdit2015Age, c, on="Country", how='outer')

dataEdit2015Age=dataEdit2015Age.drop(columns=["population_x","suicides_no_y"])

dataEdit2015Age.rename(columns={"suicides_no_x":"suicides","population_y":"Total Population"},inplace=True)

dataEdit2015Age = dataEdit2015Age.loc[dataEdit2015Age["Total Population"]>0]

dataEdit2015Age["ratio"]=dataEdit2015Age["suicides"]/dataEdit2015Age["Total Population"]*100000

age_order=["5-14 years","15-24 years","25-34 years","35-54 years","55-74 years","75+ years"]

correlationToRatioGroup(data=dataEdit2015Age,dataHDI=HDI2015,orders=age_order,group="age")
factors = ["Country",'Employment in agriculture (% of total employment) 2010-2014','Employment in services (% of total employment) 2010- 2014','Unemployment Youth (% ages 15-24) 2010-2014','Unemployment Youth not in school or employment (% ages 15-24) 2010-2014',]

age_order=["5-14 years","15-24 years","25-34 years","35-54 years","55-74 years","75+ years"]        

HDIAge = dataHDI[factors]

dataEdit2010_2014 = dataEdit.loc[dataEdit["year"].isin(["2010","2011","2012","2013","2014"])]

dataEdit2010_2014Age=dataEdit2010_2014.groupby(["Country","year","age"]).agg({"suicides_no":"sum","population":"sum"}).reset_index()

dataEdit2010_2014Age

dataEdit2010_2014Age["ratio"]=dataEdit2010_2014Age["suicides_no"]/dataEdit2010_2014Age["population"]*100000



dataEdit2010_2014Age=dataEdit2010_2014Age.groupby(["Country","age"]).agg({"ratio":"mean"}).reset_index()

correlationToRatioGroup(data=dataEdit2010_2014Age,dataHDI=HDIAge,orders=age_order,group="age")