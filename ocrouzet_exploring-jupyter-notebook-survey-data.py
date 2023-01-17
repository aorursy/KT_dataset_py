%matplotlib inline

from IPython.display import HTML
import pandas as pd

survey = pd.read_csv("../input/Survey.csv")
unique_values = pd.melt(survey, var_name="Column", value_name="UniqueResponses").groupby("Column")["UniqueResponses"].nunique()

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
sns.countplot(data=survey, x="Status")
HTML("%d people started the survey, and %d people completed it" % (len(survey), len([s for s in survey["Status"] if s=="Complete"])))

completed = survey.iloc[[s=="Complete" for s in survey["Status"]],:]

from textwrap import wrap
for col in [col for col in unique_values[unique_values<10].index if col!="Status"]:
    plt.figure()
    ax=sns.countplot(data=completed, x=col)
    ax.set_xlabel("\n".join(wrap(col, width=70)))

from wordcloud import WordCloud
from matplotlib._pylab_helpers import Gcf
from IPython.core.pylabtools import print_figure 
from base64 import b64encode

def wordcloud(text):
    cloud = WordCloud(max_font_size=80, 
                      relative_scaling=.5,
                      background_color="white",
                      width=750,
                      height=250).generate(text)
    plt.figure(figsize=(15,5))
    plt.imshow(cloud)
    plt.axis("off")
    # plt.show()
    # hacky way to show HTML+Image ... http://stackoverflow.com/questions/28877752/ipython-notebook-how-to-combine-html-output-and-matplotlib-figures
    fig = Gcf.get_all_fig_managers()[-1].canvas.figure
    image_data = "<img src='data:image/png;base64,%s'>" % b64encode(print_figure(fig)).decode("utf-8")
    Gcf.destroy_fig(fig)
    return image_data

html = []   

for col in [col for col in unique_values[unique_values>10].index if col not in ["Time Started", "Date Submitted"]]:
    if "#2" in col or "#3" in col: # don't have time to properly combine these
        continue
    html.append("<h3>" + col + "</h3>")
    html.append(completed[col].value_counts()
                              .sort(ascending=False, inplace=False)[:5]
                              .to_frame(name="Number of Responses")
                              .reset_index(level=0, inplace=False)
                              .rename(columns={"index":"Response Text"})
                              .to_html())
    html.append(wordcloud(" ".join([x for x in completed[col] if type(x)==type("")])))
HTML("\n".join(html))

HTML(unique_values.to_frame().to_html())

HTML(survey.head().to_html())

