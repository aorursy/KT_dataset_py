%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

plt.style.use('ggplot')

pd.options.display.float_format = "{:.2f}".format # two decimal places



### helper functions ##

flatten = lambda l: [item for sublist in l for item in sublist]  ## takes a list(list) -> list





def createDummies(column, _data):

    _data[column].fillna("", inplace=True)

    uniqueVals = set(flatten(list(map(lambda str: str.split("; "),

                                      list(_data[column].unique())))))

    uniqueVals.remove("")



    _dummies = pd.DataFrame(0, index=_data.index, columns=uniqueVals)

    for dummyCol in _dummies.columns:

        dummies = _data[column].map(lambda row: dummyCol in row)

        _dummies[dummyCol] = dummies



    return _dummies, uniqueVals





############# Preprocess Data #############

### Read data ###

so_data = pd.read_csv("../input/survey_results_public.csv")



### common column names and variables ###

devType = "DeveloperType"

salary = "Salary"

dScien = "Data scientist"

experience = "YearsProgram"

language = "HaveWorkedLanguage"

jobT = "Job Title"

years = "YearsProgram"

yearsN = "YearsProgramNumeric"

ide = "IDE"

stats_long_name = 'Developer with a statistics or mathematics background'

stats_short_name = "Stats/Math Background"

emb_long_name = 'Embedded applications/devices developer'

emb_short_name = "Embedded Systems"



job_dummies, jobTitles = createDummies(devType, so_data)

language_dummies, languageTitles = createDummies(language, so_data)



year_conversions = {"Less than a year": 0,

                    "1 to 2 years": 2,

                    "2 to 3 years": 3,

                    "3 to 4 years": 4,

                    "4 to 5 years": 5,

                    "5 to 6 years": 6,

                    "6 to 7 years": 7,

                    "7 to 8 years": 7,

                    "8 to 9 years": 9,

                    "9 to 10 years": 10,

                    "10 to 11 years": 11,

                    "11 to 12 years": 12,

                    "12 to 13 years": 13,

                    "13 to 14 years": 14,

                    "14 to 15 years": 15,

                    "15 to 16 years": 16,

                    "16 to 17 years": 17,

                    "17 to 18 years": 18,

                    "18 to 19 years": 19,

                    "20 or more years": 20

                    }

years_numeric = so_data[years].map(year_conversions)

years_numeric.name = yearsN

so_data = so_data.join(years_numeric)
sal_v_type = pd.melt(so_data.join(job_dummies).loc[:, [salary] + list(jobTitles)], id_vars=salary).dropna()

sal_v_type = sal_v_type[sal_v_type["value"]]

del sal_v_type["value"]

sal_v_type.columns = [salary, jobT]



sal_v_type.loc[sal_v_type[jobT] == stats_long_name, jobT] = stats_short_name

sal_v_type.loc[sal_v_type[jobT] == emb_long_name, jobT] = emb_short_name



output = sal_v_type.groupby(jobT)[salary].mean().sort_values(ascending=False).to_frame()

output.columns = ["Average Salary"]

output
g = sns.FacetGrid(sal_v_type, col=jobT, col_wrap=4, sharey=False)

g = g.map(plt.hist, salary)
diff_in_salary = so_data.join(job_dummies).groupby([years, dScien])[salary].mean()

gain = (diff_in_salary.shift(-1) - diff_in_salary)[::2].reset_index(1, drop=True).sort_values(ascending=False).to_frame()

gain.columns = ["Average gain in salary: average(DS) - average(All other groups)"]

gain
joiners = [salary, yearsN]

sal_exp_v_type = pd.melt(so_data.join(job_dummies).loc[:, joiners + list(jobTitles)],

                         id_vars=joiners).dropna()

sal_exp_v_type = sal_exp_v_type[sal_exp_v_type["value"]]

del sal_exp_v_type["value"]

sal_exp_v_type.columns = [salary, yearsN, jobT]



sal_exp_v_type.loc[sal_exp_v_type[jobT] == stats_long_name, jobT] = stats_short_name

sal_exp_v_type.loc[sal_exp_v_type[jobT] == emb_long_name, jobT] = emb_short_name



sal_exp_v_type[yearsN] = sal_exp_v_type[yearsN].astype(int)



# In general it seems being a Data Scientist, machine learning eingeering, or stats/math programmer increases your pay

sns.factorplot(x="YearsProgramNumeric", y="Salary", col="Job Title", data=sal_exp_v_type, col_wrap=4).savefig(

    "stuff.png")
datascience_data = so_data[so_data.loc[:, devType].str.contains(dScien, na=False)]



dscien_langauge_dummies, dscien_lang = createDummies(language, datascience_data)

ds_lang = (dscien_langauge_dummies.sum(0).sort_values(ascending=False) / len(dscien_langauge_dummies) * 100.).to_frame()

ds_lang.columns = ["% Data Scientists reported working with the language in the last year"]

ds_lang



_langauge_dummies, _lang = createDummies(language, so_data)

_lang = (_langauge_dummies.sum(0) / len(_langauge_dummies) * 100).sort_values(ascending=False).to_frame()

_lang.columns = ["% ALL USERS reported working with the language in the last year"]

_lang
rName = "r_only"

pName = "python_only"

bothName = "python_and_r"

r_only = _langauge_dummies["R"] & (~_langauge_dummies["Python"])

r_only.name = rName

python_only = (_langauge_dummies["Python"]) & (~_langauge_dummies["R"])

python_only.name = pName

python_and_r = (_langauge_dummies["Python"]) & (_langauge_dummies["R"])

python_and_r.name = bothName



s_ds_lang = so_data.join([r_only, python_only, python_and_r]).loc[:, [salary, yearsN, rName, pName, bothName]].melt(

    id_vars=[salary, yearsN])

s_ds_lang = s_ds_lang[s_ds_lang["value"]]

del s_ds_lang["value"]

s_ds_lang.columns = [salary, "Years Programming", "Type"]

output = s_ds_lang.groupby("Type")[salary].mean().sort_values(ascending=False).to_frame()

output.columns = ["Mean Salary"]

output
sns.boxplot(x="Type", y=salary, data=s_ds_lang)
s_ds_lang.groupby("Type")["Years Programming"].mean().sort_values(ascending=False).to_frame()
sns.boxplot(x="Type", y="Years Programming", data=s_ds_lang)
dscien_ide_dummies, ides = createDummies(ide, datascience_data)

output = (dscien_ide_dummies.sum(0) / len(dscien_ide_dummies) * 100).sort_values(ascending=False).to_frame()

output.columns = ["% Data Scientists regularly use IDE"]

output
_ide_dummies, ides = createDummies(ide, so_data)

output = (_ide_dummies.sum(0) / len(_ide_dummies) * 100).sort_values(ascending=False).to_frame()

output.columns = ["% ALL USERS regularly use IDE"]

output
joinerz = list(ides) + [yearsN]

ide_v_yearsN = so_data.join(_ide_dummies).loc[:, joinerz].melt(id_vars=yearsN)

ide_v_yearsN = ide_v_yearsN[ide_v_yearsN["value"]]

del ide_v_yearsN["value"]

ide_v_yearsN.columns = [yearsN, ide]



popular_ides = list(_ide_dummies.sum().sort_values(ascending=False)[0:11].index)



ide_across_years = pd.crosstab(index=ide_v_yearsN[yearsN], columns=ide_v_yearsN[ide])

ide_across_years = ide_across_years.div(ide_across_years.sum(1), axis=0)

popular_ides_v_years = ide_across_years[popular_ides]



handlez = plt.plot(popular_ides_v_years)

plt.title("% IDE use by years programming")

plt.legend(handles=handlez, labels=popular_ides, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)