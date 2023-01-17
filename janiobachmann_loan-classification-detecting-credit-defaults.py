# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Statistical Libraries
from scipy.stats import norm
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from scipy import stats
import warnings
% matplotlib inline
from plotly import tools
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import warnings
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
# Import the datasets
train = pd.read_csv("../input/application_train.csv")
test = pd.read_csv("../input/application_test.csv")
pos_cash = pd.read_csv("../input/POS_CASH_balance.csv")
bureau_bal = pd.read_csv("../input/bureau_balance.csv")
bureau = pd.read_csv("../input/bureau_balance.csv")
previous_app = pd.read_csv("../input/bureau.csv")
installment_payment = pd.read_csv("../input/installments_payments.csv")
credit_card_bal = pd.read_csv("../input/credit_card_balance.csv")
plt.style.use('ggplot')
# Merge train and test for data analysis
frames = [train, test]
complete_df = pd.concat(frames, sort=True)

f, axes = plt.subplots(ncols=4, figsize=(20,5)) 

# Filter Nas
annuity_dist = complete_df["AMT_ANNUITY"].dropna()

# Distributions 
sns.distplot(complete_df["AMT_CREDIT"], kde=True, color="#DF3A01", ax=axes[0], fit=norm).set_title("Amount Credit Distribution")
axes[0].set_xlabel("Amount Credit")
axes[0].set_xticklabels(complete_df["AMT_CREDIT"], rotation=45)

sns.distplot(annuity_dist, kde=True, color="#DF3A01", ax=axes[1], fit=norm).set_title("Amount Credit Distribution")
axes[1].set_xlabel("Amount Annuity")
axes[1].set_xticklabels(annuity_dist, rotation=45)

# Let's use this using log (Without log it is heavily skewed)
log_income = np.around(np.log(complete_df["AMT_INCOME_TOTAL"]),2)

sns.distplot(log_income, kde=True, color="#DF3A01", ax=axes[2], fit=norm).set_title("Income Total")
axes[2].set_xlabel("Income")
axes[2].set_xticklabels(log_income, rotation=45)

# Amount Goods Price
price_withoutna = complete_df["AMT_GOODS_PRICE"].dropna()

sns.distplot(price_withoutna, kde=True, color="#DF3A01", ax=axes[3], fit=norm).set_title("Amount of Goods Price")
axes[3].set_xlabel("Goods Price")
axes[3].set_xticklabels(price_withoutna, rotation=45)
f, axes = plt.subplots(ncols=4, figsize=(20,5)) 

# Use the norms
stats.probplot(complete_df["AMT_CREDIT"], plot=axes[0])
axes[0].set_title("Amount Credit \n Probability Plot")
stats.probplot(annuity_dist, plot=axes[1])
axes[1].set_title("Annuity Amount \n Probability Plot")
stats.probplot(log_income, plot=axes[2])
axes[2].set_title("Income \n Probability Plot")
stats.probplot(price_withoutna, plot=axes[3])
axes[3].set_title("Goods Price \n Probability Plot")
f, axes = plt.subplots(ncols=4, figsize=(20,5)) 


# Distributions 
sns.distplot(np.log(complete_df["AMT_CREDIT"]), kde=True, color="#DF3A01", ax=axes[0], fit=norm).set_title("Amount Credit Distribution")
axes[0].set_xlabel("Amount Credit")
axes[0].set_xticklabels(complete_df["AMT_CREDIT"], rotation=45)

sns.distplot(np.log(annuity_dist), kde=True, color="#DF3A01", ax=axes[1], fit=norm).set_title("Amount Credit Distribution")
axes[1].set_xlabel("Annuity Amount")
axes[1].set_xticklabels(annuity_dist, rotation=45)

# Let's use this using log (Without log it is heavily skewed)
log_income = np.around(np.log(complete_df["AMT_INCOME_TOTAL"]),2)

sns.distplot(log_income, kde=True, color="#DF3A01", ax=axes[2], fit=norm).set_title("Income Total")
axes[2].set_xlabel("Income")
axes[2].set_xticklabels(log_income, rotation=45)

# Amount Goods Price
price_withoutna = complete_df["AMT_GOODS_PRICE"].dropna()

sns.distplot(np.log(price_withoutna), kde=True, color="#DF3A01", ax=axes[3], fit=norm).set_title("Amount of Goods Price")
axes[3].set_xlabel("Goods Price")
axes[3].set_xticklabels(price_withoutna, rotation=45)
f, axes = plt.subplots(ncols=4, figsize=(20,5)) 

stats.probplot(np.log(complete_df["AMT_CREDIT"]), plot=axes[0])
axes[0].set_title("Amount Credit \n Probability Plot")
stats.probplot(np.log(annuity_dist), plot=axes[1])
axes[1].set_title("Annuity Amount \n Probability Plot")
stats.probplot(np.log(log_income), plot=axes[2])
axes[2].set_title("Income \n Probability Plot")
stats.probplot(np.log(price_withoutna), plot=axes[3])
axes[3].set_title("Goods Price \n Probability Plot")
plt.style.use('dark_background')
fig = plt.figure(figsize=(16,8))

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(212)

# Create 2 subplots above and one down for distribution by Gender

# First Subplot
main_genders = complete_df.loc[(complete_df['CODE_GENDER'] == 'M') | (complete_df['CODE_GENDER'] == 'F')]

gender_housing = main_genders.groupby(['CODE_GENDER', 'NAME_HOUSING_TYPE']).size()
gender_housing.unstack().plot(kind='bar', stacked=True, colormap='RdBu', figsize=(16,10), grid=False, ax=ax1)
ax1.set_xlabel("Genders")
ax1.set_ylabel("Count")
ax1.set_title("Which Housing Type did Genders Bought?")
ax1.legend(loc="best", prop={'size': 10})

# Second Subplot
gender_income = main_genders.groupby(['CODE_GENDER', 'NAME_INCOME_TYPE']).size()
gender_income.unstack().plot(kind='barh', stacked=True, colormap='RdBu', figsize=(16,10), grid=False, ax=ax2)
ax2.set_xlabel("Count")
ax2.set_ylabel("Genders")
ax2.set_title("Income Type by Gender")
ax2.legend(loc="best", prop={'size': 10})


# Third Subplot (Income and Credit scatterplot by Gender )
# Sample df
sampled_data = main_genders.sample(frac=1)
sampled_data = sampled_data[:200]
sns.regplot(x=sampled_data["AMT_CREDIT"], y=sampled_data["AMT_INCOME_TOTAL"], color="r", ax=ax3)
ax3.set_title("Positive Correlation between Credit and Income")
# Let's Change the Style
plt.style.use('classic')


# Continue the Gender Analysis: (Distribution by Gender we need two subplots)
f, axes = plt.subplots(2, 2, figsize=(20,12)) 


# Separate Genders
male = main_genders.loc[main_genders["CODE_GENDER"] == "M"]
female = main_genders.loc[main_genders["CODE_GENDER"] == "F"]

# Male Distribution of Credit
sns.distplot(male["AMT_CREDIT"], kde=True, color="#FA5858", ax=axes[0][0], fit=norm).set_title("Male Credit Distribution")
axes[0][0].set_xlim(0,)
axes[0][0].set_facecolor("#F2F5A9")

# Female Distribution of Credit
sns.distplot(female["AMT_CREDIT"], kde=True, color="#FA5858", ax=axes[1][0], fit=norm).set_title("Female Credit Distribution")
axes[1][0].set_xlim(0,)
axes[1][0].set_facecolor("#F2F5A9")

# # Income Distribution by Gender
log_male = np.around(np.log(male["AMT_INCOME_TOTAL"]),2)
sns.distplot(log_male, kde=True, color="#58FA58", ax=axes[0][1], fit=norm).set_title("Male Inccome Distribution")
axes[0][1].set_facecolor("#F2F5A9")

log_female = np.around(np.log(female["AMT_INCOME_TOTAL"]),2)
sns.distplot(log_female, kde=True, color="#58FA58", ax=axes[1][1], fit=norm).set_title("Female Inccome Distribution")
axes[1][1].set_facecolor("#F2F5A9")


plt.show()
plt.style.use('seaborn-whitegrid')
plt.clf()

sample_genders = main_genders[:1000]
shuffled_sample = sample_genders.sample(frac=1)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))

# Income Distribution
sns.boxplot(x="CODE_GENDER", y="AMT_INCOME_TOTAL", data=shuffled_sample, ax=axes[0], palette="RdBu")
axes[0].set_title("Walls Material and an Individuals Income")

# Credit Distribution
sns.boxplot(x="CODE_GENDER", y="AMT_CREDIT", data=shuffled_sample, ax=axes[1], palette="RdBu")
axes[1].set_title("Walls Material and an Individuals Credit")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
plt.suptitle('Relationship between Saleprice \n and Categorical Utilities', fontsize=12)
sns.pointplot(x='NAME_EDUCATION_TYPE', y='AMT_INCOME_TOTAL', hue='CODE_GENDER', data=shuffled_sample, ax=axes[0])
axes[0].set_xticklabels(shuffled_sample["NAME_EDUCATION_TYPE"], rotation=90)
sns.pointplot(x='NAME_CONTRACT_TYPE', y='AMT_INCOME_TOTAL', hue='CODE_GENDER', data=shuffled_sample, ax=axes[1])
axes[1].set_xticklabels(shuffled_sample["NAME_CONTRACT_TYPE"], rotation=90)

plt.legend(loc='best')
plt.show()
# Material Analysis
material_df = complete_df.dropna(subset=['WALLSMATERIAL_MODE'])
# sample_df = material_gender[:1000]
# sample_df = sample_df.sample(frac=1)


material_df["WALLSMATERIAL_MODE"].unique()
material_df.head()
avg_credit = round(material_df.groupby(["WALLSMATERIAL_MODE"], as_index=False).AMT_CREDIT.mean(), 2)

credits_applied = avg_credit["AMT_CREDIT"].values.tolist()
types_materials = avg_credit["WALLSMATERIAL_MODE"].unique()


data = [go.Scatterpolar(
  r = credits_applied,
  theta = types_materials,
  fill = 'toself',
    line = dict(
            color = "#00BFFF"
        ),
        mode = 'lines+markers',
        marker = dict(
            color = '#FE2E2E',
            symbol='square',
            size=6
        )
)]

layout = go.Layout(
    title= "Amount of Credit by Type of Material",
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [min(credits_applied), max(credits_applied)]
    )
  ),
  showlegend = False,
    paper_bgcolor = "#F3F781"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename = "radar/basic")
# Let's go see if there is a similar pattern with Annuity, Goods_Price and Income Total
# Is Monolisthic the highest for those three features

amt_annuity = material_df.groupby(["WALLSMATERIAL_MODE"], as_index=False).AMT_ANNUITY.mean()
amt_goods_price = material_df.groupby(["WALLSMATERIAL_MODE"], as_index=False).AMT_GOODS_PRICE.mean()
amt_income = material_df.groupby(["WALLSMATERIAL_MODE"], as_index=False).AMT_INCOME_TOTAL.mean()



# Radar Charts (Three of them) with the distribution per Month of Sales


data = [
    go.Scatterpolar(
        mode='lines+markers',
        r = amt_annuity["AMT_ANNUITY"].values.tolist(),
        theta = amt_annuity["WALLSMATERIAL_MODE"].unique(),
        fill = 'toself',
        name="Annuity",
        line=dict(
            color="rgba(255,215,0, 0.95)" 
        ),
        marker=dict(
            color="#FE2E2E",
            symbol="square",
            size=8
        ),
        subplot = "polar"
    ),
    go.Scatterpolar(
        mode='lines+markers',
        r = amt_goods_price["AMT_GOODS_PRICE"].values.tolist(),
        theta = amt_goods_price["WALLSMATERIAL_MODE"].unique(),
        fill = 'toself',
        name="Goods Prices",
        line=dict(
            color="rgba(0,255,127, 0.95)" 
        ),
        marker=dict(
            color="#FE2E2E",
            symbol="square",
            size=8
        ),
        subplot = "polar2"
    ),
    go.Scatterpolar(
        mode='lines+markers',
        r = amt_income["AMT_INCOME_TOTAL"],
        theta = amt_income["WALLSMATERIAL_MODE"].unique(),
        fill = 'toself',
        name="Income",
        line=dict(
            color="rgba(165,42,42, 0.95)"
        ),
        marker=dict(
            color="#FE2E2E",
            symbol="square",
            size=8
        ),
        subplot = "polar3"
    )
]

layout = go.Layout(
    title="Average Annuities, Goods Price and Income <br> <i> by Types of Material </i>",
    showlegend = False,
     paper_bgcolor = "rgb(255,248,220)",
    polar = dict(
      domain = dict(
        x = [0,0.3],
        y = [0,1]
      ),
      radialaxis = dict(
        tickfont = dict(
          size = 8
        )
      ),
      angularaxis = dict(
        tickfont = dict(
          size = 8
        ),
        rotation = 90,
        direction = "counterclockwise"
      )
    ),
    polar2 = dict(
      domain = dict(
        x = [0.35,0.65],
        y = [0,1]
      ),
      radialaxis = dict(
        tickfont = dict(
          size = 8
        )
      ),
      angularaxis = dict(
        tickfont = dict(
          size = 8
        ),
        rotation = 85,
        direction = "clockwise"
      ),
    ),
    polar3 = dict(
      domain = dict(
        x = [0.7, 1],
        y = [0,1]
      ),
      radialaxis = dict(
        tickfont = dict(
          size = 8
        )
      ),
      angularaxis = dict(
        tickfont = dict(
          size = 8
        ),
        rotation = 90,
        direction = "clockwise"
      ),
    ))

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='polar/directions')
# Styling Tables official documentation from Pandas

def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: rgb(60,179,113)' if v else '' for v in is_max]


cross_types = pd.crosstab(material_df["WALLSMATERIAL_MODE"], material_df["NAME_HOUSING_TYPE"])



cross_types.style.apply(highlight_max)
cm = sns.light_palette("seagreen", as_cmap=True)

cross_types.style.background_gradient(cmap=cm)