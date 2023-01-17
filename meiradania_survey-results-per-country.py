import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df = pd.read_csv("../input/multipleChoiceResponses.csv", encoding="latin-1", low_memory=False)

df["CompensationAmountFloat"] = df.CompensationAmount.map(lambda x: float(str(x).replace(",","").replace("-", "-1")))

df["CompensationAmountFloatClean"] = df["CompensationAmountFloat"].map(lambda x: np.where(x>0,x,None))



summary = df[["CompensationAmountFloatClean"]].describe()

summary
df_country = df[df.Country.notnull()]

print("Number of Total Respondents: {0:,}".format(df_country.EmploymentStatus.count()))
df_country.groupby(["Country"])["EmploymentStatus"].count().sort_values().plot(kind="barh",

                                                                               title="Number of Respondents per Country",

                                                                              figsize=(18,12));
def is_outlier(value, p25, p75):

    """Check if value is an outlier

    """

    lower = p25 - 1.5 * (p75 - p25)

    upper = p75 + 1.5 * (p75 - p25)

    return value <= lower or value >= upper





def plot_hbar(df, column, ax, title="Current Job Title"):

    b = df.groupby(column).EmploymentStatus.count().plot(kind='barh', 

                                                             ax=ax,  

                                                             title=title)

    

    b.set_ylabel("")

    b.set_xlabel("Respondents")

    return b

    

def stats_per_country(country, compensation_currency):

    

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20,30))

    fig.tight_layout() 

    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=1.5, hspace=0.5)

    df_ = df_country[df_country.Country == country]

    print("Country Respondents: {0} ({1:.1%} of total respondents)".format(df_.EmploymentStatus.count(),

                                                 float(df_.EmploymentStatus.count())/df.EmploymentStatus.count()))

    

    a = df_.groupby("Age").EmploymentStatus.count().plot(ax=axes[0,0], title="Age")

    a.set_ylabel("Respondents")



    df_["CompensationAmountFloatIsOutlier"] = df_["CompensationAmountFloatClean"].map(lambda x: 

                                                                                is_outlier(x, summary.loc['25%'].item(), summary.loc['75%'].item()))



    df_noOutlier = df_[df_.CompensationAmountFloatIsOutlier == False]

    df_noOutlier['CompensationAmountGroup'] = pd.cut(df_noOutlier.CompensationAmountFloatClean, bins=10)

    compensation_summary = df_noOutlier.groupby(['CompensationAmountGroup']).CompensationAmountFloatClean.describe()

    c = compensation_summary.reset_index().plot(kind="barh",

                                            x="CompensationAmountGroup", y="count",

                                            ax=axes[0,1], 

                                            title="Compensation in {0}".format(compensation_currency))

    c.set_xlabel("Respondents")

    

    es = plot_hbar(df=df_, column="EmploymentStatus", ax=axes[1,0], title="Employment Status")

    cjt = plot_hbar(df=df_, column="CurrentJobTitleSelect", ax=axes[1,1], title="Current Job Title")

    cjt = plot_hbar(df=df_, column="FormalEducation", ax=axes[2,0], title="Formal Education")

    cjt = plot_hbar(df=df_, column="MajorSelect", ax=axes[2,1], title="Major Select")

    

    temp = df_noOutlier[df_noOutlier.CompensationCurrency==compensation_currency].groupby(["Age","CompensationAmountFloatClean"]).EmploymentStatus.count().reset_index()

    temp["Factor"] =  temp.CompensationAmountFloatClean * temp.EmploymentStatus

    avg_compensation_per_age = temp.groupby("Age").agg({"Factor": np.mean})

    avg_compensation_per_age.plot(ax=axes[3,0], title="Average Compensation per Age", legend=None)
stats_per_country("Germany", compensation_currency="EUR")
stats_per_country("Brazil", compensation_currency="BRL")