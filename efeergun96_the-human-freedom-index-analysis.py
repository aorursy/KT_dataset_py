import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

OriginalDataSet = pd.read_csv("../input/hfi_cc_2018.csv")
OriginalDataSet.info()  # more columns than I desired to work for
OriginalDataSet.head(3)  # Looking to the Columns
SimplerDataSet = pd.DataFrame() # creating a empty dataset

SimplerDataSet["pf_rule_of_law"] = OriginalDataSet.pf_rol            # Rule Of Law
SimplerDataSet["pf_security_safety"] = OriginalDataSet.pf_ss         # Security and Safety
SimplerDataSet["pf_movement"] = OriginalDataSet.pf_movement          # Movement
SimplerDataSet["pf_religion"] = OriginalDataSet.pf_religion          # Religion
SimplerDataSet["pf_association"] = OriginalDataSet.pf_association    # Association, Assembly, and Civil Society
SimplerDataSet["pf_expression"] = OriginalDataSet.pf_expression      # Expression and Information
SimplerDataSet["pf_identity"] = OriginalDataSet.pf_identity          # Identity and Relationships
SimplerDataSet["ef_government"] = OriginalDataSet.ef_government      # Size of Government
SimplerDataSet["ef_legal"] = OriginalDataSet.ef_legal                # Legal System and Property Rights
SimplerDataSet["ef_money_access"] = OriginalDataSet.ef_money         # Access to Sound Money
SimplerDataSet["ef_trade"] = OriginalDataSet.ef_trade                # Freedom to Trade Internationally
SimplerDataSet["ef_regulation"] = OriginalDataSet.ef_regulation      # Regulation of Credit, Labor, and Business

SimplerDataSet["country"] = OriginalDataSet.countries                # Name of the Country
SimplerDataSet["year"] = OriginalDataSet.year                        # Year of Observation
SimplerDataSet["eco_free_score"] = OriginalDataSet.ef_score          # Economical Freedom Score
SimplerDataSet["eco_free_rank"] = OriginalDataSet.ef_rank            # Economical Freedom Rank
SimplerDataSet["free_score"] = OriginalDataSet.hf_score              # Human Freedom Score
SimplerDataSet["free_rank"] = OriginalDataSet.hf_rank                # Human Freedom Rank
SimplerDataSet.head()  # first few rows
SimplerDataSet.info()
SimplerDataSet.describe()  # statitical description of new dataset
plt.figure(figsize=(14,10))
plt.title("CORRELATION HEATMAP",fontsize=20)
sns.heatmap(data=SimplerDataSet.corr(),cmap="PRGn_r",annot=True, fmt='.2f', linewidths=1)
plt.show()
freedom_class = SimplerDataSet.free_score.round(decimals=0)
economy_class = SimplerDataSet.eco_free_score.round(decimals=0)

plt.figure(figsize=(28,28))

plt.suptitle("StoryTeller Scatter Plots",fontsize=20)

plt.subplot(2,2,1)
sns.scatterplot(data=SimplerDataSet,x="pf_rule_of_law",y="ef_legal",hue=freedom_class, palette="RdYlGn",alpha=0.7,size=economy_class, sizes=(10,200))
plt.xlabel("Rule Of Law")
plt.ylabel("Legal System and Property Rights")
plt.grid()

plt.subplot(2,2,2)
sns.scatterplot(data=SimplerDataSet,x="ef_trade",y="ef_money_access",hue=freedom_class, palette="RdYlGn",alpha=0.7,size=economy_class, sizes=(10,200))
plt.xlabel("Freedom to Trade Internationally")
plt.ylabel("Access to Sound Money")
plt.grid()

plt.subplot(2,2,3)
sns.scatterplot(data=SimplerDataSet,x="pf_security_safety",y="pf_expression",hue=freedom_class, palette="RdYlGn",alpha=0.7,size=economy_class, sizes=(10,200))
plt.xlabel("Security and Safety")
plt.ylabel("Expression and Information")
plt.grid()

plt.subplot(2,2,4)
sns.scatterplot(data=SimplerDataSet,x="pf_association",y="ef_regulation",hue=freedom_class, palette="RdYlGn",alpha=0.7,size=economy_class, sizes=(10,200))
plt.xlabel("Association, Assembly, and Civil Society")
plt.ylabel("Regulation of Credit, Labor, and Business")
plt.grid()

plt.show()
Best_3_free = SimplerDataSet[SimplerDataSet["free_rank"] <= 5]
Best_3_eco = SimplerDataSet[SimplerDataSet["eco_free_rank"] <= 5]

Worst_3_free = SimplerDataSet[SimplerDataSet["free_rank"] >= 158]
Worst_3_eco = SimplerDataSet[SimplerDataSet["eco_free_rank"] >= 158]
plt.figure(figsize=(20,16))

plt.suptitle("Amount Of Top 5 and Bottom 5 Appearencies",fontsize=16)

plt.subplot(2,2,1)
plt.title("Top 5 Appearencies on Freedom Rank")
Best_3_free.country.value_counts().plot.bar()


plt.subplot(2,2,2)
plt.title("Top 5 Appearencies on Economical Freedom Rank")
Best_3_eco.country.value_counts().plot.bar()


plt.subplot(2,2,3)
plt.title("Worst 5 Appearencies on Freedom Rank")
Worst_3_free.country.value_counts().plot.bar()


plt.subplot(2,2,4)
plt.title("Worst 5 Appearencies on Economical Freedom Rank")
Worst_3_eco.country.value_counts().plot.bar()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
