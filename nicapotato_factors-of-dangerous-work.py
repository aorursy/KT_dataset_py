from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
# General Imports
import numpy as np
import pandas as pd

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Load
df = pd.read_csv("../input/severeinjury.csv", encoding="latin-1")

## Processing
# Drop Unwanted
df.drop(["ID","UPA","Address1","Address2","Primary NAICS","Inspection",
         "Nature","Part of Body","Event","Source","Secondary Source"],axis=1,inplace=True)

# Change Dtypes
df.Hospitalized = df.Hospitalized.astype(int)
df.Amputation= df.Amputation.fillna(0).astype(int)
df['EventDate'] = pd.to_datetime(df["EventDate"])

# Capitalize Category Values
for x in ["City","State"]:
    df[x] = df[x].str.capitalize()

# Clean Employer Variable
df["Employer"] = df["Employer"].str.capitalize().str.replace('[^\w\s]','')
df.loc[df.Employer.str.contains(r"(?i)Postal Service|(?i)States Postal|(?i)USPS"),"Employer"] = "US postal service"
df.loc[df.Employer.str.contains(r"\b(?i)ups\b"),"Employer"] = 'United parcel service'
# Remove company type from name.
df["Employer"] = df.Employer.str.replace(r"\binc\b|\bllc\b|\blde\b|\bco\b|\bcorp\b|\bllp\b", '').str.strip()

"""
Regular Expressions:
I noticed that companie names were recorded in multiple different ways. This is an attempt to unify these inconsistencies.

Regular Expression Explanation:
- [^\w\s] : [^] Negatve Set \w Words \s White Spaces
- (?i)ups : Removes case sensitivity from ups -> UPS, UPs, ups etc. would be returned
- r"\bups\b" : In this case "ups" will only be returned if it borders (b) a non-letter. e.i its a exact word.
- | : Represents "or" argument

Creating the Gender Variable:
"""

# New Variables
df["Male"] = False
df["Female"] = False

# True when gender pronoun appears
df.loc[df['Final Narrative'].str.contains(
    r'(?i)\bhis\b|(?i)\bhe\b|(?i)\bmale\b|(?i)\bhim\b'),["Male"]] = True
df.loc[df['Final Narrative'].str.contains(
    r'(?i)\bhers\b|(?i)\bshe\b|(?i)\bfemale\b|(?i)\bher\b'),["Female"]] = True

# Gender Var
df["Gender"] = "NULL"
df.loc[(df.Male == True),"Gender"] = "Male"
df.loc[(df.Female == True),"Gender"] = "Female"
df.loc[(df.Female == True)&(df.Male == True),"Gender"] = "Both Mentioned"

"""
Hospitalized/Amputated - Binary Variable
"""

# Fast funcdtion for top occruence categories
def topcat_index(series, n=5):
    """
    Wow! 2 charcters SAVED on function length
    """
    return series.value_counts().index[:n]
def topcats(series, n=5):
    return series.isin(topcat_index(series, n=n))

## New Amputation Variable
df["Amputated"]= False
df.loc[df["Amputation"] > 0,"Amputated"] = True

# New Hospital
df["Hospital"]= False
df.loc[df["Hospitalized"] > 0,"Hospital"] = True

print("Pre-Processing Complete")

print("Click 'Show Output' to see re Gender Variable:")

for x in ["Male","Female","Gender"]:
    print("\n{}:\n".format(x))
    print(df[x].value_counts())
    print(df[x].value_counts(normalize=True))

print("\n{} % o both gender pronouns are precent".format(
    round((df[(df.Male == True) & (df.Female == True)].shape[0]/df.shape[0])*100,2)))
def custom_describe(df):
    """
    I am a non-comformist :)
    """
    unique_count = []
    for x in df.columns:
        mode = df[x].mode().iloc[0]
        unique_count.append([x,
                             len(df[x].unique()),
                             df[x].isnull().sum(),
                             mode,
                             df[x][df[x]==mode].count(),
                             df[x].dtypes])
    print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))
    return pd.DataFrame(unique_count, columns=["Column","Unique","Missing","Mode","Mode Occurence","dtype"]).set_index("Column").T

custom_describe(df)
def time_slicer(df, timeframes):
    """
    Function to count observation occurrence through different lenses of time.
    """
    f, ax = plt.subplots(len(timeframes), figsize = [12,10])
    for i,x in enumerate(timeframes):
        df.loc[:,[x,"Employer"]].groupby([x]).count().plot(ax=ax[i],color="purple")
        ax[i].set_ylabel("Incident Count")
        ax[i].set_title("Incident Count by {}".format(x))
        ax[i].set_xlabel("")
    ax[len(timeframes)-1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)
    
# Time Frames of Interest
df["Date of Year"] = df['EventDate'].dt.dayofyear # Day of Year
df["Weekday"] = df['EventDate'].dt.weekday 
df["Quarter"] = df['EventDate'].dt.quarter
df["Day of Month"] = df['EventDate'].dt.day

# Plot
time_slicer(df=df,timeframes=["EventDate","Date of Year","Quarter","Day of Month","Weekday"])
with sns.color_palette("viridis"):
    f, ax = plt.subplots(1,2, figsize=[10,4],sharey=True)
    sns.countplot(df["Hospitalized"],ax=ax[0])
    ax[0].set_title("Distribution of Hospitalized")
    ax[0].set_ylabel("Count")
    sns.countplot(df["Amputation"],ax=ax[1])
    ax[1].set_title("Distribution of Amputation")
    ax[1].set_ylabel("")
    plt.tight_layout(pad=0)
def cat_countplot(df, var, size = [8,10], n=15):
    """
    Function to plot multiple seaborn count plots horizontally, and ordered by most frequent class.
    """
    with sns.color_palette("inferno"):
        f, ax = plt.subplots(len(var),figsize=size)
        for i,x in enumerate(var):
            temptop = df[x].value_counts().index[:n]
            sns.countplot(y=df[x][df[x].isin(temptop)], order=temptop, ax=ax[i],palette='viridis')
            ax[i].set_title("Distribution of {}".format(x))
            ax[i].set_xlabel("")
        ax[len(var)-1].set_xlabel("Count")
        plt.tight_layout(pad=0)

def continous_plots(df, var):
    """
    Built this and found it doesn't apply to my problem at hand.
    """
    f, ax = plt.subplots(1,len(var),figsize=(12,4), sharey=False)
    for i,x in enumerate(var):
        sns.distplot(df[x], ax=ax[i])
        ax[i].set_title("{} Distribution".format(x))
        ax[i].set_ylabel("")
        ax[i].set_xlabel("Distribution")
    ax[0].set_ylabel("Density")
    plt.tight_layout()
    
# Plot
with sns.color_palette("inferno"):
    cat_countplot(df=df, var=["Employer","City","State", "Gender"],n=20, size=[14,13])
cat_countplot(df=df, var=["NatureTitle","SourceTitle","Secondary Source Title","Part of Body Title"],n=20,size=[14,14])
with sns.plotting_context("notebook", font_scale=1.7):
    f, ax = plt.subplots(1,figsize=[14,10])
    x= "EventTitle"
    n=30
    temptop = df[x].value_counts().index[:n]

    sns.countplot(y=df[x][df[x].isin(temptop)], order=temptop, ax=ax, palette="viridis")
    ax.set_title("Distribution of {}".format(x))
    ax.set_ylabel("")
plt.show()
for x in ["EventTitle", "SourceTitle"]:
    print("{} Class Count: {}".format(x,len(set(df[x]))))
    print("{} STD: {}\n".format(x,df[x].value_counts().std()))
sns.heatmap(pd.crosstab(df['Amputated'], df["Hospital"], normalize=True).mul(100).round(0),
            annot=True, linewidths=.5,fmt='g', cmap="viridis",
                cbar_kws={'label': 'Percentage %'})
plt.title("Hospital and Amputated Table")
plt.show()
col = "Part of Body Title"
temp = df.loc[(topcats(df["Employer"]))
              &(topcats(df[col],n=4))
              ,:]
sns.factorplot(y="Employer", hue="Amputated", col=col, col_wrap=2,
            data=temp, kind="count", palette='viridis',
               order=df["Employer"].value_counts().index[:5],aspect=2)
plt.show()
def target_rate_by(df, target, by):
    """
    So I can look at amputated (target) rate through different lenses (by)
    """
    amp_rates = (pd.crosstab(df[by], df[target], normalize = 'index')
                 .mul(100).round(0).reset_index())
    amp_rates.columns = [by, "False Rate","Rate"]
    amp_rates["Count"] = "NaN"
    
    for x in set(amp_rates[by]):
        amp_rates.loc[amp_rates[by]==x, "Count"] = df.loc[df[by]==x, by].count()

    return (amp_rates[[by,"Rate", "Count"]]
     .sort_values(by="Rate",ascending=False))

variables = ["EventTitle","SourceTitle","NatureTitle","Part of Body Title",
             "Employer","State","City","Gender"]
top = 10
limit = 7

# Plot
f, ax = plt.subplots(len(variables),figsize=[8,25])
for i,x in enumerate(variables):
    temp = target_rate_by(df=df, target="Amputated", by=x) # Custom Function
    sns.barplot(x="Rate",y=x, ax=ax[i],palette='viridis',
                data= temp.loc[(temp.Count > limit)][:top])
    ax[i].set_title("{} Variable".format(x))
    ax[i].set_xlabel("")
    ax[i].set_ylabel("")
ax[len(variables)-1].set_xlabel("Amputation Rate")
#plt.tight_layout(pad=0)
plt.subplots_adjust(top=0.95)
plt.suptitle('Amputation Rate by Categories - Top {} Subclass with Incident Count Over {}'
             .format(top,limit),fontsize=17)
plt.show()
def cat_time_slicer(df, slicevar, n, timeframes):
    """
    Function to count observation occurrence through different lenses of time.
    """
    f, ax = plt.subplots(len(timeframes), figsize = [11,7])
    top_classes = topcat_index(df[slicevar],n=n)
    for i,x in enumerate(timeframes):
        for y in top_classes:
            total = df.loc[df[slicevar]==y,slicevar].count()
            ((df.loc[(df[slicevar]==y),[x,"Employer"]]
             .groupby([x])
             .count()/total).fillna(0)
            .plot(ax=ax[i], label=y))
        ax[i].set_ylabel("Percent of\nCompany Incidents")
        ax[i].set_title("Percent of Incident by Company by {}".format(x))
        ax[i].set_xlabel("")
        ax[i].legend(top_classes, fontsize='large', loc='center left',bbox_to_anchor=(1, 0.5))
    ax[len(timeframes)-1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)
    plt.subplots_adjust(top=0.90)
    plt.suptitle('Normalized Time-Series for top {} {}s over different over {}'.format(n,slicevar,timeframes),fontsize=17)
cat_time_slicer(df=df, slicevar = "Employer",
                n=5, timeframes=["EventDate","Date of Year","Quarter","Weekday"])
cat_time_slicer(df=df, slicevar = "City",
                n=4, timeframes=["EventDate","Date of Year","Quarter","Weekday"])
cat_time_slicer(df=df, slicevar = "State",
                n=4, timeframes=["EventDate","Date of Year","Quarter","Weekday"])
cat_time_slicer(df=df, slicevar = "Amputated",
                n=4, timeframes=["EventDate","Date of Year","Quarter","Weekday"])
cat_time_slicer(df=df[df.Gender != "Both Mentioned"], slicevar = "Gender",
                n=5, timeframes=["EventDate","Date of Year","Quarter","Weekday"])
[print("{}\n".format(x)) for x in 
 df["Final Narrative"].loc[
     (df["Amputated"]==True) & 
     (df["Final Narrative"]
      .str.contains(r"(?i)gun|(?i)burn|(?i)shock"))]
 .sample(10)]
# Topic Modeling
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from wordcloud import WordCloud, STOPWORDS

size = (10,7)
# Function
def cloud(text, title):
    # Processing Text
    stopwords = set(STOPWORDS) # Redundant
    wordcloud = WordCloud(width=800, height=400,
                          #background_color='white',
                          #stopwords=stopwords,
                         ).generate(" ".join(text))
    
    # Output Visualization
    fig = plt.figure(figsize=size, dpi=80, facecolor='k',edgecolor='k')
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=50,color='y')
    plt.tight_layout(pad=0)
    plt.show()

lemm = WordNetLemmatizer()
class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))
    
# Define helper function to print top words
def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "Topic #{}: ".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
        print(message)
    
def LDA(data, name=""):
    # Storing the entire training text in a list
    text = list(data.values)
    # Calling our overwritten Count vectorizer
    tf_vectorizer = LemmaCountVectorizer(max_df=0.95, min_df=2,
                                              stop_words='english',
                                              decode_error='ignore')
    tf = tf_vectorizer.fit_transform(text)


    lda = LatentDirichletAllocation(n_components=3, max_iter=10,
                                    learning_method = 'online',
                                    learning_offset = 50.,
                                    random_state = 0)
    lda.fit(tf)

    n_top_words = 10
    print("\n{} Topics in LDA model: ".format(name))
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)
    
# Output
pd.set_option('max_colwidth', 700)
pd.set_option('max_info_columns', 100)

def topic_view(catlevel,x,target="Final Narrative",df=df):
    df = df[df[df[catlevel]==x].notnull()]
    print(LDA(df.loc[df[catlevel]==x,target], name=x))
    cloud(df.loc[df[catlevel]==x,target].values,"{}".format(x))

    return df.loc[df[catlevel]==x,[target,"Amputation"]].sample(4)
topic_view(df=df, catlevel="Employer", x="Caterpillar")
topic_view(df=df, catlevel="Employer", x="Tyson foods")
topic_view(df=df, catlevel="Employer", x="US postal service")
topic_view(df=df, catlevel="Employer", x="Walmart")
topic_view(df=df, catlevel="Gender", x="Male")
topic_view(df=df, catlevel="Gender", x="Female")
topic_view(df=df, catlevel="Part of Body Title", x="Brain")