import pandas as pd

import numpy as np

import seaborn as sns                       #visualisation

import matplotlib.pyplot as plt             #visualisation

%matplotlib inline     

sns.set(color_codes=True)
pisa= pd.read_csv("../input/pisa-scores-2015/Pisa mean perfromance scores 2013 - 2015 Data.csv")

pisa.head()

pisa.tail()
pisa.dtypes
pisa = pisa.drop(["Country Code","2013 [YR2013]","2014 [YR2014]"],axis = 1)

pisa.head(5)
pisa = pisa.rename(columns = {"Series Name":"Score Name","Series Code":"Score Code","2015 [YR2015]":2015})

pisa = pisa.set_index("Country Name")

pisa.head(5)
pisa = pisa.drop_duplicates()

pisa.head(9)
pisa.count()
def convert_to_float(row):

    if row == "..":

        return row

    else:

        return float(row)

pisa[2015] = pisa[2015].apply(convert_to_float)
pisa_math_male = pisa.loc[(pisa["Score Code"] == "LO.PISA.MAT.MA")]

pisa_math_male = pisa_math_male.loc[pisa_math_male[2015] != ".."]

pisa_math_male = pisa_math_male.sort_values(by = 2015) #sort values by score

mean_data_male = pisa_math_male[2015].mean()

pisa_math_male.tail()
pisa_math_male[2015].plot(kind = "barh",figsize=(8,20))



plt.bar(mean_data_male,100,color = "black",width= 4)

plt.yticks(size = 11)

plt.xlabel("Scores",size = 15)

plt.ylabel("Country Name",size = 15)

plt.title("Pisa Math Scores(male) by Country",size = 20)

plt.show()
pisa_math_female = pisa.loc[(pisa["Score Code"] == "LO.PISA.MAT.FE")]

pisa_math_female = pisa_math_female.loc[pisa_math_female[2015] != ".."]

pisa_math_female = pisa_math_female.sort_values(by = 2015) #sort values by score

pisa_math_female_mean = pisa_math_female[2015].mean()

pisa_math_female.tail()
pisa_math_female[2015].plot(kind = "barh",figsize=(8,20),color = "green")



plt.bar(pisa_math_female_mean,100,color = "red",width= 4)

plt.yticks(size = 11)

plt.xlabel("Scores",size = 15)

plt.ylabel("Country Name",size = 15)

plt.title("Pisa Math Scores(Female) by Country",size = 20)

plt.show()
pisa_math_female_male = pd.merge(pisa_math_male,pisa_math_female,left_index = True,right_index = True,how = "left")

pisa_math_female_male = pisa_math_female_male.rename(columns = {"2015_x":"2015 Male","2015_y" : "2015 Female"})

pisa_math_female_male = pisa_math_female_male.drop(['Score Name_x',"Score Name_y","Score Code_x","Score Code_y"],axis = 1)

pisa_math_female_male_difference = ((pisa_math_female_male["2015 Female"] - pisa_math_female_male["2015 Male"])/pisa_math_female_male["2015 Male"])*100

pisa_math_female_male_difference = pisa_math_female_male_difference.sort_values()

pisa_math_female_male_difference.tail()
pisa_math_female_male_difference.plot(kind = "barh",figsize = (8,20))

plt.bar(0,120,width = 0.1,color = "black")

plt.title("How much better are females at Math",size = 20)

plt.yticks(size = 11)

plt.xlabel("% difference from male",size = 15)

plt.ylabel("Country Name",size = 15)

plt.show()
pisa_reading_male = pisa.loc[(pisa["Score Code"] == "LO.PISA.REA.MA")]

pisa_reading_male = pisa_reading_male.loc[pisa_reading_male[2015] != ".."]

pisa_reading_male = pisa_reading_male.sort_values(by = 2015) #sort values by score

pisa_reading_male_mean = pisa_reading_male[2015].mean()

pisa_reading_male.tail()
pisa_reading_male[2015].plot(kind = "barh",figsize=(8,20))



plt.bar(pisa_reading_male_mean,100,color = "black",width= 4)

plt.yticks(size = 11)

plt.xlabel("Scores",size = 15)

plt.ylabel("Country Name",size = 15)

plt.title("Pisa Reading Scores(Male) by Country",size = 20)

plt.show()
pisa_reading_female = pisa.loc[(pisa["Score Code"] == "LO.PISA.REA.FE")]

pisa_reading_female = pisa_reading_female.loc[pisa_reading_female[2015] != ".."]

pisa_reading_female = pisa_reading_female.sort_values(by = 2015) #sort values by score

pisa_reading_female_mean = pisa_reading_female[2015].mean()

pisa_reading_female.tail()
pisa_reading_female[2015].plot(kind = "barh",figsize=(8,20),color = "green")



plt.bar(pisa_reading_female_mean,100,color = "red",width= 4)

plt.yticks(size = 11)

plt.xlabel("Scores",size = 15)

plt.ylabel("Country Name",size = 15)

plt.title("Pisa Reading Scores(Female) by Country",size = 20)

plt.show()
pisa_reading_female_male = pd.merge(pisa_reading_male,pisa_reading_female,left_index = True,right_index = True,how = "left")

pisa_reading_female_male = pisa_reading_female_male.rename(columns = {"2015_x":"2015 Male","2015_y" : "2015 Female"})

pisa_reading_female_male_difference = ((pisa_reading_female_male["2015 Female"] - pisa_reading_female_male["2015 Male"])/pisa_reading_female_male["2015 Male"])*100

pisa_reading_female_male_difference = pisa_reading_female_male_difference.sort_values()

pisa_reading_female_male_difference.tail()
pisa_reading_female_male_difference.plot(kind = "barh",figsize = (8,20))

plt.bar(0,120,width = 0.1,color = "black")

plt.title("How much better are females at Reading",size = 20)

plt.yticks(size = 11)

plt.xlabel("% difference from male",size = 15)

plt.ylabel("Country Name",size = 15)

plt.show()
pisa_science_male = pisa.loc[(pisa["Score Code"] == "LO.PISA.SCI.MA")]

pisa_science_male = pisa_science_male.loc[pisa_science_male[2015] != ".."]

pisa_science_male = pisa_science_male.sort_values(by = 2015) #sort values by score

pisa_science_male_mean = pisa_science_male[2015].mean()

pisa_science_male.tail()
pisa_science_male[2015].plot(kind = "barh",figsize=(8,20))



plt.bar(pisa_science_male_mean,100,color = "black",width= 4)

plt.yticks(size = 11)

plt.xlabel("Scores",size = 15)

plt.ylabel("Country Name",size = 15)

plt.title("Pisa Sciecne Scores(Male) by Country",size = 20)

plt.show()
pisa_science_female = pisa.loc[(pisa["Score Code"] == "LO.PISA.SCI.FE")]

pisa_science_female = pisa_science_female.loc[pisa_science_female[2015] != ".."]

pisa_science_female = pisa_science_female.sort_values(by = 2015) #sort values by score

pisa_science_female_mean = pisa_science_female[2015].mean()

pisa_science_female.tail()
pisa_science_female[2015].plot(kind = "barh",figsize=(8,20),color = "green")



plt.bar(pisa_science_female_mean,100,color = "red",width= 4)

plt.yticks(size = 11)

plt.xlabel("Scores",size = 15)

plt.ylabel("Country Name",size = 15)

plt.title("Pisa Science Scores(Female) by Country",size = 20)

plt.show()

pisa_science_female_male = pd.merge(pisa_science_male,pisa_science_female,left_index = True,right_index = True,how = "left")

pisa_science_female_male = pisa_science_female_male.rename(columns = {"2015_x":"2015 Male","2015_y" : "2015 Female"})

pisa_science_female_male_difference = ((pisa_science_female_male["2015 Female"] - pisa_science_female_male["2015 Male"])/pisa_science_female_male["2015 Male"])*100

pisa_science_female_male_difference = pisa_math_female_male_difference.sort_values()

pisa_math_female_male_difference.tail()
pisa_math_female_male_difference.plot(kind = "barh",figsize = (8,20))

plt.bar(0,120,width = 0.1,color = "black")

plt.title("How much better are females at Science",size = 20)

plt.yticks(size = 11)

plt.xlabel("% difference from male",size = 15)

plt.ylabel("Country Name",size = 15)

plt.show()
pisa_all_subjects_mean_male = pisa.loc[(pisa["Score Code"] == "LO.PISA.MAT.MA") | (pisa["Score Code"] == "LO.PISA.REA.MA") | (pisa["Score Code"] == "LO.PISA.SCI.MA") ]

pisa_all_subjects_mean_male = pisa_all_subjects_mean_male.drop(pisa.loc[pisa[2015] == ".."].index)

pisa_all_subjects_mean_male[2015] = pd.to_numeric(pisa_all_subjects_mean_male[2015])

pisa_all_subjects_mean_male = pisa_all_subjects_mean_male.groupby("Country Name").mean()

pisa_all_subjects_mean_male = pisa_all_subjects_mean_male.sort_values(by = 2015)

pisa_all_subjects_mean_male_mean = pisa_all_subjects_mean_male.mean()

pisa_all_subjects_mean_male.tail()
pisa_all_subjects_mean_male[2015].plot(kind = "barh",figsize=(8,20))



plt.bar(pisa_all_subjects_mean_male_mean,100,color = "black",width= 4)

plt.yticks(size = 11)

plt.xlabel("Scores",size = 15)

plt.ylabel("Country Name",size = 15)

plt.title("Pisa All subjects Score(male) by Country",size = 20)

plt.show()
pisa_all_subjects_mean_female = pisa.loc[(pisa["Score Code"] == "LO.PISA.MAT.FE") | (pisa["Score Code"] == "LO.PISA.REA.FE") | (pisa["Score Code"] == "LO.PISA.SCI.FE") ]

pisa_all_subjects_mean_female = pisa_all_subjects_mean_female.drop(pisa.loc[pisa[2015] == ".."].index)

pisa_all_subjects_mean_female[2015] = pd.to_numeric(pisa_all_subjects_mean_female[2015])

pisa_all_subjects_mean_female = pisa_all_subjects_mean_female.groupby("Country Name").mean()

pisa_all_subjects_mean_female = pisa_all_subjects_mean_female.sort_values(by = 2015)

pisa_all_subjects_mean_female_mean = pisa_all_subjects_mean_female.mean()

pisa_all_subjects_mean_female.tail()
pisa_all_subjects_mean_female[2015].plot(kind = "barh",figsize=(8,20))



plt.bar(pisa_all_subjects_mean_female_mean,100,color = "black",width= 4)

plt.yticks(size = 11)

plt.xlabel("Scores",size = 15)

plt.ylabel("Country Name",size = 15)

plt.title("Pisa All subjects Score(female) by Country",size = 20)

plt.show()
pisa_math_female_male_difference = pd.DataFrame(pisa_math_female_male_difference,columns = {"math"})

pisa_reading_female_male_difference = pd.DataFrame(pisa_reading_female_male_difference,columns = {"reading"})

pisa_science_female_male_difference = pd.DataFrame(pisa_science_female_male_difference,columns = {"science"})

pisa_all_subjects_female_male = pd.merge(pisa_math_female_male_difference,pisa_reading_female_male_difference,left_index = True,right_index = True,how = "left")

pisa_all_subjects_female_male = pd.merge(pisa_all_subjects_female_male,pisa_science_female_male_difference,left_index = True,right_index = True,how = "left")

pisa_all_subjects_female_male = (pisa_all_subjects_female_male.sum(axis = 1)/3).sort_values()

pisa_all_subjects_female_male.tail()
pisa_all_subjects_female_male.plot(kind = "barh",figsize=(8,20))



plt.bar(0,100,color = "black",width= 0.09)

plt.yticks(size = 11)

plt.xlabel("Scores",size = 15)

plt.ylabel("Country Name",size = 15)

plt.title("How much better are females at All subjects",size = 20)

plt.show()