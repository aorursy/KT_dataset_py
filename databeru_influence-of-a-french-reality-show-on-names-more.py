import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/french-baby-names/national_names.csv")
df.head(5)
df.describe()
sns.heatmap(df.isnull())
plt.title("Missing values?")
plt.show()
count_by_year = pd.pivot_table(df, index = "year", values = "count", aggfunc=np.sum).reset_index()
plt.figure(figsize = (12,6))
plt.fill_between(count_by_year["year"],count_by_year["count"], lw = 5)
plt.ylim(0,max(count_by_year["count"]+100000))
plt.xlim(min(count_by_year["year"]), max(count_by_year["year"])+2)
plt.title("Number of baby names by year", fontsize = 18)
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()
# Create two sets: 
#  - Male names
#  - Female names
unique_names_sets = (set(df[df["sex"] == "M"]["name"].unique()), set(df[df["sex"] == "F"]["name"].unique()))

# Number of male names, names for both genders and female names
unique_names = [len(unique_names_sets[0] - unique_names_sets[1]), 
                len(unique_names_sets[0] & unique_names_sets[1]), 
                len(unique_names_sets[1] - unique_names_sets[0])]

# Visualize the result
plt.figure(figsize=(8,5))
plt.bar(["Male","Both","Female"], unique_names, color = ["black", "cyan", "pink"])
plt.title("Number of total unique names by Gender", fontsize = 18)
plt.show()
# Calculate the number of unique names by gender and by year
count_namesF = df[df["sex"] == "F"].groupby(by = "year").count().rename(columns = {"name":"Females"})
count_names = df[df["sex"] == "M"].groupby(by = "year").count().rename(columns = {"name":"Males"})
count_names["Females"] = count_namesF["Females"]
count_names.drop(["sex","count"], axis = 1, inplace = True)
count_names.reset_index(inplace = True)

# Display the result
fig = plt.figure(figsize = (12,6))
ax = fig.add_axes([0,0,1,1])
count_names.plot(x = "year", y = "Males", color = "black", lw = 5, ax = ax)
count_names.plot(x = "year", y = "Females", color = "pink", lw = 5, ax = ax)
ax.set_title("Number of unique names by gender and by year", fontsize = 18)
plt.legend(fontsize = 15)
ax.set_xlabel("Year",fontsize = 18)
plt.show()
# Create two columns with dummies for sex (M/F)
df = pd.get_dummies(df, columns = ["sex"])

# Count the number of occurences of the names for males and females
# in the columns sex_F and sex_M
df["sex_F"] = df["count"] * df["sex_F"]
df["sex_M"] = df["count"] * df["sex_M"]
# df.drop("count", axis = 1, inplace = True)

# Keep only the names which are used for males and females
group_name = df.groupby(by = "name").sum().drop("year", axis = 1)
both_sex = group_name[(group_name["sex_F"] > 0) & (group_name["sex_M"] > 0)].copy()

# Calculate the proportion of males/females having each name
both_sex["proportion_M/F"] = both_sex["sex_M"] / both_sex["sex_F"]

# Keep only names with a M/F proportion between 0.95 and 1.05
# Keep only names which are present at least 41 times for girls
gender_neutral = both_sex[(both_sex["proportion_M/F"] > 0.95) & (both_sex["proportion_M/F"] < 1.05) & (both_sex["sex_F"] > 40)]
from wordcloud import WordCloud

def display_word(words, title = None):
# Display the words in the string "text" as word cloud

    wordcloud = WordCloud(
            background_color='white',
            max_font_size=20, 
            scale=3,
            random_state=0 # chosen at random by flipping a coin; it was heads
    ).generate(str(words))
    fig = plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.title(title, fontsize = 40)
    plt.show()

words = " ".join(gender_neutral.index)
display_word(words)
# Names with a M/F proportion between 0.95 and 1.05
gender_neutral
feminine_names = both_sex.sort_values(by = "proportion_M/F", ascending = True).drop("proportion_M/F", axis = 1)[:15]
words = " ".join(feminine_names.index)
display_word(words)
feminine_names
manly_names = both_sex.sort_values(by = "proportion_M/F", ascending = False).drop("proportion_M/F", axis = 1)[:15]
words = " ".join(manly_names.index)
display_word(words)
manly_names
fig = plt.figure(figsize = (12,6))
ax = fig.add_axes([0,0,1,1])
ax.axvline(2001, color='r', lw = 3, ls = "--", c = "black")
df[df["name"] == "Loana"][10:].plot(x = "year", y = "count", ax = ax, lw = 5)
plt.title("Number of kids called Loana by year", fontsize = 18)
ax.set_xlabel('Loft Story (2001)', position=(0.155, 2e6), horizontalalignment='left', fontsize = 15)
plt.show()
fig = plt.figure(figsize = (12,6))
ax = fig.add_axes([0,0,1,1])
ax.axvline(2001, color='r', lw = 3, ls = "--", c = "black")
df[df["name"] == "Steevy"][10:].plot(x = "year", y = "count", ax = ax, lw = 5)
plt.title("Number of kids called Steevy by year", fontsize = 18)
ax.set_xlabel('Loft Story (2001)', position=(0.33, 2e6), horizontalalignment='left', fontsize = 15)
plt.show()
from collections import Counter

# Count the number of occurrences of each letter in the names
evg = df["name"]*df["count"]
evg = "".join(evg).lower()
letters_freq_name = Counter(evg)
# The frequency of letters in french
letters_freq = letters_freq=[['e',115024205],['a',67563628],['i',62672992],
['s',61882785],['n',60728196],['r',57656209],['t',56267109],['o',47724400],
['l',47171247],['u',42698875],['d',34914685],['c',30219574],['m',24894034],
['p',23647179],['g',11684140],['b',10817171],['v',10590858],['h',10583562],
['f',10579192],['q',6140307],['y',4351953],['x',3588990],['j',3276064],
['k',2747547],['w',1653435],['z',1433913]]

letters_freq = pd.DataFrame(letters_freq, columns=["letter","count"])
letters_freq = letters_freq.sort_values("letter")[:26].reset_index(drop=True)
letters_freq["freq_french"] = letters_freq["count"] / letters_freq["count"].sum()
letters_freq.drop(columns = "count", inplace = True)

# The frequency of letters in the names
let = "abcdefghijklmnopqrstuvwxyz"
lst = []
for k in letters_freq_name.keys():
    if k in let:
        lst.append([k, letters_freq_name[k]])
freq_name = pd.DataFrame(lst, columns = ["letter", "count"]).sort_values("letter").reset_index(drop=True)

freq_name["freq_in_names"] = freq_name["count"] / freq_name["count"].sum()

# Combine the frequency of letters in French and the frequency of letters in the names
letters_freq["freq_in_names"] = freq_name["freq_in_names"]
labels = letters_freq.letter.values

width = 0.35
x = np.arange(len(labels))
fig, ax = plt.subplots(1,1, figsize = (15,8))
ax.bar(x - width/2, letters_freq.freq_french.values, width, label='French corpus', color = "black")
ax.bar(x + width/2, letters_freq.freq_in_names.values, width, label='French names', color = "pink")

ax.set_ylabel('')
ax.set_title('Frequency of letters\nin names VS in French corpus', fontsize = 20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize = 18)
ax.legend()

fig.tight_layout()

plt.show()
letters_freq["difference"] = letters_freq["freq_in_names"] / letters_freq["freq_french"]

colors = letters_freq["difference"].apply(lambda x: "FUCHSIA" if x > 1.5 else ("silver" if x > 0.67 else "purple")).values

fig, ax = plt.subplots(1,1, figsize = (15,8))
ax.bar(x = letters_freq["letter"], height = letters_freq["difference"], color = colors)
ax.hlines(y = 1, xmin = -1, xmax = 26, color = "black", lw = 3)
ax.set_xticklabels(labels, fontsize = 18)
plt.title("Most/Less used letters in names\nin comparison to French corpus", fontsize = 20)
plt.text(-2, 0.85, "Igual\nuse", fontsize = 18)
plt.show()