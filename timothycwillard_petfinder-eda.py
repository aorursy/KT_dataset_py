import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
from scipy.stats import ks_2samp
from sklearn.feature_selection import chi2
import json
import matplotlib.image as mpimg
from scipy.stats import linregress
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif

def kappa_score(y1, y2, labels=None, sample_weight=None):
    return cohen_kappa_score(y1, y2, labels=labels, weights="quadratic", sample_weight=sample_weight)
kappa_scorer = make_scorer(kappa_score)

sns.set(style="darkgrid", context="notebook")
xsize = 12.0
ysize = 8.0

rand_seed = 24680
np.random.seed(rand_seed)

import os
print(os.listdir("../input"))
print(os.listdir("../input/train"))
print(os.listdir("../input/test"))
train_df = pd.read_csv("../input/train/train.csv")
test_df = pd.read_csv("../input/test/test.csv")
train_df.info()
test_df.info()
train_df.head()
test_df.head()
X_cols = ["Type", "Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize", 
          "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "Fee", "State", 
          "VideoAmt", "PhotoAmt"]
fig, axes = plt.subplots(ncols=2, nrows=len(X_cols))
fig.set_size_inches(2.0*xsize, len(X_cols)*ysize)
axes = axes.flatten()

for i, col in enumerate(X_cols):
    ix = 2*i
    ixx = ix + 1
    sns.countplot(train_df[col], ax=axes[ix])
    axes[ix].set_title("Train "+col+" Feature Distribution")
    sns.countplot(test_df[col], ax=axes[ixx])
    axes[ixx].set_title("Test "+col+" Feature Distribution")

plt.show()
count_cols = ["Type", "Gender", "Color1", "Color2", "Color3", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health"]
fig, axes = plt.subplots(nrows=len(count_cols))
fig.set_size_inches(xsize, len(count_cols)*ysize)
axes = axes.flatten()

for i, col in enumerate(count_cols):
    sns.countplot(x=col, hue="AdoptionSpeed", data=train_df, ax=axes[i])
    axes[i].set_title(col+" Feature Count by AdoptionSpeed")

plt.show()
reg_cols = ["Age", "Quantity", "Fee", "VideoAmt", "PhotoAmt"]
fig, axes = plt.subplots(nrows=len(reg_cols), ncols=2)
fig.set_size_inches(2.0*xsize, len(reg_cols)*ysize)
axes = axes.flatten()

y = train_df["AdoptionSpeed"].values
for i, col in enumerate(reg_cols):
    ix = 2*i
    ixx = ix+1
    
    x = train_df[col].values
    logx = np.log(1.0 + x)
    
    m, b, r, pval, stderr = linregress(x=x, y=y)
    axes[ix].plot(x, y, "o")
    axes[ix].plot(x, m*x + b, "-", label=r"$r^2=%.4f$"%(r**2.0)+"\n"+r"$pval=%.4f$"%pval)
    axes[ix].legend()
    axes[ix].set_xlabel(col)
    axes[ix].set_ylabel("AdoptionSpeed")
    
    m, b, r, pval, stderr = linregress(x=logx, y=y)
    axes[ixx].plot(logx, y, "o")
    axes[ixx].plot(logx, m*logx + b, "-", label=r"$r^2=%.4f$"%(r**2.0)+"\n"+r"$pval=%.4f$"%pval)
    axes[ixx].legend()
    axes[ixx].set_xlabel("Log "+col)
    axes[ixx].set_ylabel("AdoptionSpeed")

plt.show()
def highlight_pvals(val):
    if val < 0.01:
        return "background-color: #99ff99"
    elif val < 0.05:
        return "background-color: #ffff99"
    else:
        return "background-color: #ff9999"
ks_stat = np.zeros(len(X_cols))
ks_pval = np.zeros(len(X_cols))

for i, col in enumerate(X_cols):
    ks_stat[i], ks_pval[i] = ks_2samp(train_df[col], test_df[col])

ks_test_df = pd.DataFrame(data={"Feature":X_cols, "KS Statistic":ks_stat, "P-Value":ks_pval})
ks_test_df.style.applymap(highlight_pvals, subset=["P-Value"])
chi_stat = np.zeros(len(X_cols))
chi_pval = np.zeros(len(X_cols))

for i, col in enumerate(X_cols):
    chi_stat[i], chi_pval[i] = chi2(train_df[col].values.reshape(-1, 1), train_df["AdoptionSpeed"].values)

chi_test_df = pd.DataFrame(data={"Feature":X_cols, "Chi^2 Statistic":chi_stat, "P-Value":chi_pval})
chi_test_df.style.applymap(highlight_pvals, subset=["P-Value"])
print(os.listdir("../input/train_images")[:12])
imgs = []
for i, file in enumerate(os.listdir("../input/train_images")[:12]):
    imgs.append(mpimg.imread("../input/train_images/"+file))
fig, axes = plt.subplots(ncols=4, nrows=3)
fig.set_size_inches(24.0, 24.0)
axes = axes.flatten()

for i in range(12):
    axes[i].imshow(imgs[i])

plt.show()
train_df["PetID"].values[:12]
with open("../input/train_sentiment/86e1089a3.json", "r") as file:
    sentiment = json.load(file)
print(json.dumps(sentiment, indent=2))
sentence = ""
for obj in sentiment["sentences"]:
    sentence += " "+obj["text"]["content"]
print(sentence)
pet_ids = train_df["PetID"].values
sentiment_magnitude = np.zeros(len(pet_ids))
sentiment_score = np.zeros(len(pet_ids))
missing_count = 0

for i, pet_id in enumerate(pet_ids):
    try:
        with open("../input/train_sentiment/"+pet_id+".json", "r") as file:
            sentiment = json.load(file)
        sentiment_magnitude[i] = sentiment["documentSentiment"]["magnitude"]
        sentiment_score[i] = sentiment["documentSentiment"]["score"]
    except FileNotFoundError:
        missing_count += 1
        sentiment_magnitude[i] = np.nan
        sentiment_score[i] = np.nan

train_df["SentimentMagnitude"] = sentiment_magnitude
train_df["SentimentScore"] = sentiment_score
train_df["LogSentimentMagnitude"] = np.log(1.0 + sentiment_magnitude)
train_df["LogSentimentScore"] = np.log(1.0 + sentiment_score)
fig, axes = plt.subplots(ncols=2, nrows=2)
fig.set_size_inches(2.0*xsize, 2.0*ysize)
axes = axes.flatten()

sns.regplot(x="SentimentMagnitude", y="AdoptionSpeed", data=train_df, ax=axes[0])

sns.regplot(x="SentimentScore", y="AdoptionSpeed", data=train_df, ax=axes[1])

sns.regplot(x="LogSentimentMagnitude", y="AdoptionSpeed", data=train_df, ax=axes[2])

sns.regplot(x="LogSentimentScore", y="AdoptionSpeed", data=train_df, ax=axes[3])

plt.show()
pet_ids = train_df["PetID"].values
sentiment_sentences = []
missing_count = 0

for i, pet_id in enumerate(pet_ids):
    try:
        with open("../input/train_sentiment/"+pet_id+".json", "r") as file:
            sentiment = json.load(file)
        sentence = ""
        for obj in sentiment["sentences"]:
            sentence += " "+obj["text"]["content"]
        sentiment_sentences.append(sentence)
    except FileNotFoundError:
        missing_count += 1
        sentiment_sentences.append("")

print(missing_count)
sentiment_sentences[:3]
tfidf_vectorizer = TfidfVectorizer()
tfidf_train = tfidf_vectorizer.fit_transform(sentiment_sentences).todense()
tfidf_train
F, pvals = f_classif(tfidf_train, train_df["AdoptionSpeed"].values)
fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(2.0*xsize, ysize)

sns.distplot(F, ax=axes[0], kde=False)

sns.distplot(pvals, ax=axes[1], kde=False)
axes[1].axvline(x=0.05, linestyle=":", color="k")
axes[1].axvline(x=0.01, linestyle="--", color="k")

plt.show()
print("Statistically Significant: "+str("%.3f"%(100.0*(len(pvals[pvals<0.01])/len(pvals))))+"%")
print("Statistically Ambiguous: "+str("%.3f"%(100.0*(len(pvals[(pvals>0.01)&(pvals<0.05)])/len(pvals))))+"%")
print("Not Statistically Significant: "+str("%.3f"%(100.0*(len(pvals[pvals>0.05])/len(pvals))))+"%")
