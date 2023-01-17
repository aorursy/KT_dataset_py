# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML, Javascript
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

sns.set_style('whitegrid')
# Any results you write to the current directory are saved as output.
Qnumbers = ['Q'+str(i) for i in range(1,51)]
print(Qnumbers)
qas = pd.read_csv('../input/multipleChoiceResponses.csv')
splitters = [' - Selected Choice - ', ' (include text response) ', ' (Answers must add up to 100%) ', 
            ' (Select all that apply) ']
def foo(x):
    for splitter in splitters:
        if splitter in qas.loc[0,x]:
            return(qas.loc[0,x].split(splitter)[1])

responses_cols_dict = {}
for qnum in Qnumbers:
    responses_cols_dict[qnum] = {  x:foo(x) for x in list(qas.columns[1:]) if 
                                                     ('Part' in x) & (qnum==x.split('_')[0]) }


questions1_used = ['Q'+str(i) for i in [11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 27, 28, 29, 30, 37, 38, 49]]
questions2_used = ['Q'+str(i) for i in [2, 4, 5, 6, 7, 9]]
age_map = {
    '18-21':'18-24',
    '21-24':'18-24',
    '25-29':'25-34',
    '30-34':'25-34',
    '35-39':'35-44',
    '40-44':'35-44',
    '45-49':'45-54',
    '50-54':'45-54',
    '55-59':'55-69',
    '60-69':'55-69',
    '70-79':'70+',
    '80+':'70+'
}
country_map = {'United States of America': 'United States of America',
 'Indonesia': 'Other',
 'India': 'India',
 'Colombia': 'Other',
 'Chile': 'Other',
 'Turkey': 'Other',
 'Hungary': 'Other',
 'Ireland': 'Other',
 'France': 'Other',
 'Argentina': 'Other',
 'Japan': 'Other',
 'Nigeria': 'Other',
 'Spain': 'Other',
 'Other': 'Other',
 'Iran, Islamic Republic of...': 'Other',
 'United Kingdom of Great Britain and Northern Ireland': 'Other',
 'Poland': 'Other',
 'Kenya': 'Other',
 'Denmark': 'Other',
 'Netherlands': 'Other',
 'China': 'China',
 'Sweden': 'Other',
 'Ukraine': 'Other',
 'Canada': 'Other',
 'Australia': 'Other',
 'Russia': 'Russia',
 'Austria': 'Other',
 'Italy': 'Other',
 'Mexico': 'Other',
 'Germany': 'Other',
 'I do not wish to disclose my location': 'Other',
 'Singapore': 'Other',
 'Brazil': 'Brazil',
 'Switzerland': 'Other',
 'South Africa': 'Other',
 'South Korea': 'Other',
 'Malaysia': 'Other',
 'Hong Kong (S.A.R.)': 'Other',
 'Egypt': 'Other',
 'Tunisia': 'Other',
 'Portugal': 'Other',
 'Thailand': 'Other',
 'Morocco': 'Other',
 'Pakistan': 'Other',
 'Czech Republic': 'Other',
 'Romania': 'Other',
 'Israel': 'Other',
 'Philippines': 'Other',
 'Bangladesh': 'Other',
 'Belarus': 'Other',
 'Viet Nam': 'Other',
 'Belgium': 'Other',
 'New Zealand': 'Other',
 'Norway': 'Other',
 'Finland': 'Other',
 'Greece': 'Other',
 'Peru': 'Other',
 'Republic of Korea': 'Other'}



edu_map = {'Doctoral degree': 'Doctoral degree',
 'Bachelor’s degree': 'Bachelor’s degree',
 'Master’s degree': 'Master’s degree',
 'Professional degree': 'Professional degree',
 'Some college/university study without earning a bachelor’s degree': 'College w/o degree',
 'I prefer not to answer': 'No Answer',
 'No formal education past high school': 'High School'}

salary_map = {
 '0-10,000': '0-10,000',
 '10-20,000': '10-20,000',
 '20-30,000': '20-40,000',
 '30-40,000': '20-40,000',
 '40-50,000': '40-70,000',
 '50-60,000': '40-70,000',
 '100-125,000': '100,000+',
 '60-70,000': '40-70,000',
 '70-80,000': '70-100,000',
 '90-100,000': '70-100,000',
 '125-150,000': '100,000+',
 '80-90,000': '70-100,000',
 '150-200,000': '100,000+',
 '200-250,000': '100,000+',
 '250-300,000': '100,000+',
 '500,000+': '100,000+',
 '300-400,000': '100,000+',
 '400-500,000': '100,000+'
}
major_map  = {
    "Other": "Other",
    "Engineering (non-computer focused)": "Engineering(Non-comp)",
    "Social sciences (anthropology, psychology, sociology, etc.)": "Social sciences",
    "Mathematics or statistics": "Maths & Stats",
    "A business discipline (accounting, economics, finance, etc.)": "Business",
    "Computer science (software engineering, etc.)": "Computer science",
    "Physics or astronomy": "Physics or astronomy",
    "Information technology, networking, or system administration": "Information technology",
    "Environmental science or geology": "Environmental science",
    "Medical or life sciences (biology, chemistry, medicine, etc.)": "Medical or life sciences",
    "I never declared a major": "No Major",
    "Humanities (history, literature, philosophy, etc.)": "Humanities",
    "Fine arts or performing arts": "Fine arts"
}

experience_index = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-10', '10-15', 
                                                    '15-20', '20-25', '25-30', '30 +']
salary_index = ['0-10,000', '10-20,000', '20-40,000', '40-70,000', '70-100,000', '100,000+']
education_index = ['High School', 'College w/o degree', 'Bachelor’s degree', 'Master’s degree', 'Doctoral degree',
                  'Professional degree', "No Answer"]

qas['Q2'].iloc[1:] = qas['Q2'].iloc[1:].map(age_map)
qas['Q3'].iloc[1:] = qas['Q3'].iloc[1:].map( country_map )
qas['Q4'].iloc[1:] = qas['Q4'].iloc[1:].map( edu_map )
qas['Q5'].iloc[1:] = qas['Q5'].iloc[1:].map( major_map )
qas['Q9'].iloc[1:] = qas['Q9'].iloc[1:].map( salary_map )
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
import numpy as np
from ipywidgets.embed import embed_minimal_html
from textwrap import wrap

def wrap_labels(labels):
    return(['\n'.join(wrap(l.get_text(), 40)) for l in labels])

def wrap_labels2(labels):
    return(['\n'.join(wrap(l, 40)) for l in labels])

def plot1(Q1, Q2, nrows, ncols, name, zbreak=None, index=None):
    
    cols = list(responses_cols_dict[Q1].keys())+[Q2]
    sliced_data = qas.loc[1:,cols]
    
    new_cols = [ responses_cols_dict[Q1][key] for key in responses_cols_dict[Q1] ] + [Q2]
    
    sliced_data.columns = new_cols
    
    sliced_df = (sliced_data.groupby(Q2)[list(responses_cols_dict[Q1].values())].count()
                                .sort_index().reindex(index))
  
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize=(20,10), sharex=True, sharey=True)
    fig.suptitle(name, size='large')
    gp_df = sliced_df.sum(axis=1)
    z = 0
    for i in range(nrows):
        for j in range(ncols):
            
            if zbreak :
                z = i*(nrows - 2) + j
                if (z >= zbreak):
                    break
            else:
                z = i*(nrows - 1) + j
            
            barplt = sns.barplot(x = sliced_df.columns, y = sliced_df.iloc[z]/gp_df.iloc[z], ax=ax[i,j])
            barplt.set_xticklabels( wrap_labels(barplt.get_xticklabels()), rotation=45, ha='right')
            barplt.set_title('Total number of users in {group}: {num}'.format(group = sliced_df.index[z],
                                                                             num = gp_df.iloc[z]))
    ax[i,j].set_xticklabels( wrap_labels2(list(sliced_df.columns)), rotation=45, ha='right')
    plt.show(fig)

def plot2(Q1, Q2, name):
    
    cols = list(responses_cols_dict[Q1].keys())+[Q2]
    sliced_data = qas.loc[1:,cols]
    new_cols = [ responses_cols_dict[Q1][key] for key in responses_cols_dict[Q1] ] + [Q2]
    sliced_data.columns = new_cols
    sliced_df = sliced_data.groupby(Q2).count()
    sliced_df_sum = sliced_df.sum(axis=0).sort_values()
    plt.figure( figsize=(14, 8))
    barplt = sns.barplot(x = sliced_df_sum.index, y = sliced_df_sum.values)
    barplt.set_xticklabels( wrap_labels(barplt.get_xticklabels()), rotation=45, ha='right')
    barplt.set_title(name)


def plot3(Q1, Q2, name, index=None):
    cols = list(responses_cols_dict[Q1].keys())+[Q2]
    sliced_data = qas.loc[1:,cols]
    
    new_cols = [ responses_cols_dict[Q1][key] for key in responses_cols_dict[Q1] ] + [Q2]
    sliced_data.columns = new_cols
    sliced_df = sliced_data.groupby(Q2)[list( responses_cols_dict[Q1].values() )].count().reindex(index)
    sliced_df_sum = sliced_df.sum(axis=1)
    values = sliced_df.values/(sliced_df_sum.values.reshape(sliced_df_sum.shape[0],1)+1)
    sliced_df = pd.DataFrame(values*100, index=sliced_df.index, columns=sliced_df.columns)

    fig, ax = plt.subplots(figsize=(16,10))
    
    fig.suptitle(name, size='large')
    ax = sns.heatmap(sliced_df, annot=True, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    bx = ax.twinx()

    sliced_df.index = sliced_df_sum.values.tolist()
    bx = sns.heatmap(sliced_df, annot=True, ax=bx, cbar=False)
    bx.set_yticklabels(sliced_df_sum.values, rotation=360)
    plt.show(fig)

html = """
<div>
    <button id="first{Q}" onclick="showPlot{Q}(0, this)">Count</button>
    <button  onclick="showPlot{Q}(1, this)">Age</button>
    <button  onclick="showPlot{Q}(2, this)">Country</button>
    <button  onclick="showPlot{Q}(3, this)">Education</button>
    <button  onclick="showPlot{Q}(4, this)">Degree Major</button>
    <button  onclick="showPlot{Q}(5, this)">Role</button>
    <button  onclick="showPlot{Q}(6, this)">Industry</button>
    <button  onclick="showPlot{Q}(7, this)">Salary</button>
    <button  onclick="showPlot{Q}(8, this)">Experience</button>
    
</div>
<script>

    function showPlot{Q}(i, item){{
        
        $(item).parents().siblings().children(".output_png").hide();
        $(item).parents().siblings().children(".output_png")[i].style = "display";
        $('#{Q} p').hide();
        console.log($('#{Q} p')[i]);
        $('#{Q} p')[i].style = "display: block";
        console.log($('#{Q} p')[i]);
    }};

</script>
"""

def plot_main(Q1, varname, html_string):
    
    display( HTML(html.format(Q=Q1)))
    plot2( Q1, 'Q2', 'Distribution of {var}'.format(var = varname))
    plot1( Q1, 'Q2', 3,2, "Distribution of {var} with User's age".format(var = varname))
    plot1( Q1, 'Q3', 3,2, "Distribution of {var} with User's Country ".format(var = varname))
    plot1( Q1, 'Q4', 4,2, "Distribution of {var} with User's Education".format(var = varname), zbreak=7,
                             index = education_index)
    plot3( Q1, 'Q5', "Distribution of {var} with User's Degree Major".format(var = varname))
    plot3( Q1, 'Q6', "Distribution of {var} with User's Role".format(var = varname))
    plot3( Q1, 'Q7', "Distribution of {var} with User's Industry".format(var = varname))
    plot1( Q1, 'Q9', 3,2, "Distribution of {var} with User's Salary".format(var = varname), 
                             index = salary_index)
    plot3( Q1, 'Q8', "Distribution of {var} with User's Experience".format(var = varname), 
                              experience_index)
    display(HTML(html_string))
    display( HTML("<script> showPlot{Q}(0, $('#first{Q}')); </script>".format(Q=Q1)))
    

html_string = """<div id="Q13">
  <p><b>Count</b>: The plot above shows that jupyter is most used IDE followed by RStudio, Notepad++, Pycharm and Sublime Text. It beats all of them by a large margin. Both Jupyter and Rstudio provide excellent support for analysis and visualization which explains their large user base. Rstudio and Jupyter are mainly developed for R and Python respectively which are most preferred languages for data science. Despite being launched in 2014 well after Rsutdio(2011), Pycharm(2010) and Sublime text(2008), Jupyter project has gained large popularity among users. </p>
    <p><b>Age</b>: Jupyter it at the top as most used IDE in all age groups. The second and third position differ for different age groups. The new entrants( age: 18-24) are still testing IDEs and have roughly uniform user base across the popular IDEs. The next group( age: 25-34) is showing a slight preference for Rstudio and Notepad++ with Pycharm in close distance. The next two groups(age: 35-44, age: 45-54) clearly show preference for Rstudio and Notepad++.  The other groups(age: 55-69, age: 70+) prefer Rstudio as a second choice, though third preference is not obvious as MATLAB, Visual Studio and Notepad++ are close.</p>
    <p><b>Country</b>: Jupyter is still on the top in every country, though Rstudio and pyCharm are very close to matching its user base in the USA and China respectively. In Brazil, we see Rstudio, Notepad++, Sublime text and Visual studio code are fighting for 2nd and 3rd preference. Its noteworthy that Rstudio has a very low preference among Chinese and Russian data scientists and pyCharm is their second preference. The Spyder IDE has a sizable base among Indians.</p>
    <p><b>Education</b>: Again, we see Jupyter as a top preference among all user groups based on the level of education. As we move lower to higher levels of education the user base for RStudio is increasing while the opposite is true for pyCharm and Visual Studio. The Matlab IDE is the third preference for users with a doctoral degree. The Notepad++ has a sizable base among all user groups.</p>
    <p><b>Degree Major</b>: Jupyter is the first choice for IDE among users of all groups, though Rstudio is not behind among users from Business, Maths & Stats and Social sciences. The vast number of statistics packages in CRAN explains the large user base of RSutdio in these communities. The Second and third choice varies with the user group. The computer science and IT users prefer pyCharm and Notepad++ while Engineering(non-comp) prefer RStudio and MATLAB.</p>
    <p><b>Role</b>: Jupyter is still most loved and preferred IDE for most of the user groups except for Statisticians & Data analysts who prefer RStudio. The 2nd and 3rd choice for IDE revolves around Pycharm, Notepad++, RStudio and Visual studio code among most of the user groups. The MATLAB ide has a good presence among research-related roles.</p>
    <p><b>Industry</b>: Jupyter has overwhelming support for top IDE from all user groups. The tussle over 2nd and 3rd are dominated by mostly RStudio, Notepad++  and sometimes Pycharm. The Matlab is supported mostly by users in academic and defense role.</p>
    <p><b>Salary</b>: Once again, jupyter beats every IDE in the race for top IDE among every salary group. As the salary increases share of RStudio users is increasing. The high salary groups prefer RStudio as their second IDE while the opposite is true for Notepad++.</p>
    <p><b>Experience</b>: The plot shows that jupyter is the most used IDE among all experience groups. Among, all groups Rstudio stands out as the second choice for IDE. The Notepad++ is the next choice in every group except the two seniormost groups. The visual studio's share is increasing as the experience increases while the reverse is true for sublime text.</p>
</div>"""


plot_main("Q13", "IDEs", html_string)
html_string = """<div id="Q14">
    <p><b>Count</b>: It seems that most of the data scientists don't use a notebook platform for data analysis and visualization. Next in the line is Kaggle kernels followed by Jupyterhub/binder, Google Colab, Azure Notebook, Google cloud Datalab and others. Though Kaggle kernels(July 2017) and Google's Colab(Nov 2017) were launched later than Azure notebooks(2016) & Google cloud datalab(June 2016), both have managed to attract a larger user base.The primary reason for the success of Kaggle kernels, Jupyterhub and Colab is that they don't require a Credit card or billing registration.</p>
    <p><b>Age</b>: The share of Kaggle kernels decreases as we move towards higher age groups and same are true for Colab and Jupyterhub. It is evident from the plot that higher age groups have a small user base for all notebook platforms. The percentage of users who don't use notebooks is bigger for higher age groups.</p>
    <p><b>Country</b>: For almost every country the share of None is higher than any notebook platform barring India and Brazil. The Kaggle kernel is the top choice for notebook platform for every country except the USA where it is matched by Jupyterhub. Colab's share is low in America and China when compared to other countries.</p> 
    <p><b>Education</b>: The notebook users from all education groups prefer Kaggle platform followed by Jupyterhub and google's Colab. The users show small preference to Azure notebook and Datalab.</p>
    <p><b>Degree Major</b>: The users from most of the degree groups prefer not to use a notebook platform except for computer science who equally prefer both options Kaggle platform and not using a notebook. The second choice for a notebook platform is Jupyterhub followed by Colab in the third position. The Datalab from Google has a significant user base in fine arts group.</p>
    <p><b>Role</b>: Again, Majority prefers to not use a notebook platform barring few exceptions like users in Data journalist and Scientist role. The data journalists are more likely to use Jupyterhub and azure notebooks when compared to other roles. The first choice for notebook platform is a tie between Jupyterhub and Kaggle kernels as some roles prefer Kaggle while others prefer Jupyterhub. The third choice is google's Colab except for data engineers and journalists.</p>
    <p><b>Industry</b>: Most of the users in a group prefer not to use a notebook platform. The notebook platform users give preference to Kaggle kernel over other platforms in almost all industrial groups with few exceptions like energy/mining, student and non-profit service. The third preference goes to google Colab.</p>
    <p><b>Salary</b>: Majority of users don't use notebook platforms. The share of kaggle usersis lower in high income groups when compared to lower income groups.</p>
    <p><b>Experience</b>: The users who prefer not to use a notebook platform have a majority in all experience levels. We can see that the share of Kaggle users decreases as we move to higher experience levels while the reverse is true for None. The group with 20-25 years of experience have a significant share of azure notebooks users.</p>
</div>"""

plot_main('Q14', 'Notebook Platforms', html_string)
html_string="""<div id='Q15'>
    <p><b>Count</b>: Amazon web services leads in the share for cloud computing services. Also, a large number of users haven't used a cloud provider yet. AWS, GCP, and Azure are clearly leading the cloud computing services market. AWS(2006) was the first one to launch cloud services followed by Google(2008), IBM(2009), Microsoft(2010), and Alibaba(2016). Despite being only two years older than others, AWS has managed to stay ahead by a large margin. Though, only 2 years old Alibaba is moving fast to overtake IBM.</p>
    <p><b>Age</b>: AWS is the leader in providing cloud-based services in all age groups though users having age 18-24 years almost equally prefer google cloud and AWS. The share of Google cloud in every age group is more or less same while the share of AWS & Azure increases when we move up from the lowest age group. The share of Alibaba cloud decreases as we move to up to higher age levels.</p>
    <p><b>Country</b>: AWS is the most preferred cloud service provider for all countries except China where Alibaba is sought over the others. Alibaba has a very small presence in other countries. The USA has the lowest percentage of users not opting for cloud providers. Also, AWS margin over others in the USA indicates that the gain in cloud users percentage has been captured mostly by AWS.</p>
    <p><b>Education</b>: The distribution of cloud users is pretty much same for the major cloud providers in all groups. AWS leads the race with GCP in second place closely followed by Azure in all education groups.</p>
    <p><b>Degree Major</b>: We see that AWS is at the top of cloud providers and its user base is sometimes close(Social sciences) and even beats(Fine arts, Humanities) the nonusers percentage. GCP and Azure are the first and second runner-ups respectively. The humanities group has the highest percentage of AWS users among all the degree groups.</p>
    <p><b>Role</b>: The Students, Statisticians, Marketing analysts & not employed have a very large share (>40%) of users who don't use a cloud provider. The users with senior roles and roles related to computer science are more likely to use cloud service provider as evident from the heatmap. Database engineers prefer both AWS and Azure over GCP by a large margin. AWS is the most preferred cloud service provider.</p>
    <p><b>Industry</b>: AWS has an overwhelmingly large presence among users who work in Internet-based sales or services. The academicians and student have a high probability of not using a cloud provider when compared to other industries. AWS is the leading cloud service provider followed by GCP in some industries and Azure in others.</p>
    <p><b>Salary</b>: We can see that as the salary increases a user is more likely to opt for a cloud service provider for his needs. The share of AWS and Azure increases with rising salary while GCP's share is more or less same. Also, the margin between AWS and other service providers rises with an increase in salary.</p>
    <p><b>Experience</b>: AWS leads the race for best cloud service provider with GCP and Azure as first and second runner-up. The group with less than one year of experience have highest share among non users of cloud services when compared to other groups. The higher experience groups on an average prefer IBM cloud and Azure more than lower experience groups while opposite is true for AWS.</p>
</div>"""

plot_main('Q15', 'Cloud Providers', html_string)
html_string="""<div id="Q16">
    <p><b>Count</b>: Unlike tools and libraries, all the prominent data science languages are more than a decade old. Python was launched in 1991 while R in 2000 and SQL in the 1970s. Python is the most popular language by a large margin. SQL and R are next in line followed by C/C++, Java & others after a big dip.</p>
    <p><b>Age</b>: Python is the most preferred language by all groups. For second and third positions we have some interesting differences. The users in the youngest age group 18-24 yrs prefer C/C++ and Java. For other age groups, SQL is second most preferred language except for seniormost age group which prefers R. As we move to higher age groups the share of SQL, R and C# increases barring seniormost group where SQL decreases.</p>
    <p><b>Country</b>: Python is the most suitable language in every country group. C#/NET has a very low presence among Americans and Indians. Americans are pretty much settled with SQL as 2nd and R as 3rd most suitable language. Except for Russia and China R is the 3rd most preferred language in all groups. Matlab has a sizable presence among Chinese.</p>
    <p><b>Education</b>: Python is the first choice for data scientists in every group. SQL is in the second place for all the groups except doctoral degree. The third preference differs for each group. The share of R users increases as the level of education rises.</p>
    <p><b>Degree Major</b>: Python again leads the race for most preferred language among all groups followed by SQL in some groups and R in other. R is more popular in maths & stats, Social sciences business when compared to other groups. Java and C++/C have highest share among Computer science professionals when compared to other major groups.</p>
    <p><b>Role</b>: Python is still ruling the roost in most of the groups except Statistician & Database engineer. Both R and STATA have excellent support for statistics which explains their strong user base among statisticians. C++ and MATLAB have good support among research roles. In most of the groups 2nd and 3rd preference are divided among R and SQL, though sometimes C++, Java, MATLAB and Javascript stake their claim.</p>
    <p><b>Industry</b>: Python is sitting at the top of in every group. The 2nd and 3rd most suitable language title is divided among R and SQL with few exceptions like Defense and academics where honor goes to C++ and MATLAB. STATA finds its use in the Insurance industry. Javascript has a sizable base in the Online service industry.</p>
    <p><b>Salary</b>: Python is the most preferred language in all groups, though its share decreases as we move to higher income groups. Next in line is SQL followed by R in all groups except the lowest salary band where Java and C++ also have a sizable base.</p>
    <p><b>Experience</b>: Python is the most preferred language and by a large margin in all experience levels, though its share is decreasing as we move towards higher experience levels. C#'s share shows an increasing trend as we go up in experience levels. The second most preferred language is SQL followed by R.</p>
</div>"""

plot_main('Q16', 'Languages', html_string)
html_string = """<div id="Q19"> 
    <p><b>Count</b>: Scikit-Learn is the most popular package for machine learning framework followed by Tensorflow and Keras. Scikit and Caret are old horses launched in 2007 and still going strong. Moving a bit further in the timeline we have H2O(2011), Apache spark (2012), Caffe(2014) and xgboost(2014). Except Caffe all other are general machine learning frameworks which could be one of the reasons why it has the slowest growth among these packages. In the next few years we have Keras(2015), Tensorflow(2015), CNTK(2016), PyTorch(2016), Lightgbm(April 2017), and Fastai(nov 2017). Tensorflow is the fastest gainer while CNTK is slowest.</p>
    <p><b>Age</b>: Scikit is the most preferred framework for machine learning followed by Tensorflow and Keras in all age groups except 70+ where Tensorflow beats scikit by a small margin. The 25-34 age group is the biggest user of machine learning frameworks. Pytorch share decreases as we move higher age groups while the reverse is true for randomForest</p>
    <p><b>Country</b>: Scikit is the first choice for ML framework in all countries except China where Tensorflow takes the lead. Catboost has the largest user base among Russians when compared to other countries which is logical since Catboost was developed by Yandex a Russian company. Also, Caret is least popular among Russians and Chinese probably due to low user base for R language as seen in earlier plots. Indians and Americans have the highest number of machine learning framework users.</p>
    <p><b>Education </b>: The top three positions are pretty much fixed in all groups with Scikit at the top followed by Tensorflow and Keras. The Master's degree holders are heaviest users of ml frameworks followed by Bachelors and PhDs.</p>
    <p><b>Degree Major</b>: Scikit leads the race for popular ML framework. The business, stats, and social sciences have a large share of randomForest users. Engineering(comp and non-comp) are the top users of machine learning frameworks. </p>
    <p><b>Role</b>: The salespersons have a large number of users among them who prefer not to use any ML framework. The users in analyst(business, data, and marketing) roles prefer randomForest after scikit. Scikit is still at the top and 2nd and 3rd place is occupied by Tensorflow and Keras in most of the groups. The students and data scientists are the topmost users of machine learning frameworks.</p>
    <p><b>Industry</b>: The top three positions are pretty much fixed in all groups with Scikit at the top followed by Tensorflow and Keras. The people working in Computer/Technology and Academia heavily use machine learning packages in their work.</p>
    <p><b>Salary</b>: We have the same three frameworks in the top positions and in the same order. The share of scikit decreases as we move to higher salary bands while the reverse is true for xgboost.</p>
    <p><b>Experience</b>: Again Scikit, Tensorflow and Keras occupy the top positions and in the same order. It is interesting to note that scikit and xgboost share decreases as we move towards higher experience levels. There is general downward trend in number of ml framework users as we move towards higher experience levels with group 5-10 being an exception.</p>
</div>"""

plot_main('Q19', 'ML Frameworks', html_string)
html_string="""<div id="Q21">
    <p><b>Count</b>: Matplotlib is as overwhelmingly popular visualization library followed by ggplot2 and seaborn which have the almost same share of users. Both Matplotlib(2008) and ggplot2(2008) are a decade old packages. Matplotlib is a python based library and ggplot2 is a visualization package for R language users. In the next batch, we have D3(2011), Seaborn(2012), Plotly(2012) and shiny(2012). Though Plotly, shiny and Seaborn were launched after D3 their user base is larger than D3. This could be possible because D3 is Javascript based library and javascript is not very popular among data practitioners as seen in earlier plots.</p>
    <p><b>Age</b>: The top 3 choices for Visualization are same for every age group with Matplotlib at the top followed by ggplot2 and seaborn. The user base of ggplot2 is increasing as we move to towards higher age bands while the reverse is true for Plotly and Seaborn.</p>
    <p><b>Country</b>: Matplotlib is the most popular visualization library in all countries especially China where is mighty popular. We see that pgplot2 has a low user base in Russia and China which is perfectly logical given low R user base in these countries. The share of Plotly is also very low in China when compared to other countries.</p>
    <p><b>Education</b>: The margin between matplotlib and ggplot2 decreases as we move towards higher education levels because of the decrease in matplotlib users and the increase in ggplot2 users. The Masters. Bachelors and PhDs constitute the major part of visualization packages and libraries users.</p>
    <p><b>Degree Major</b>: This plot is consistent with language plot. The groups which have good R user base are more likely to use GGplot2 and the same is true for python and Matplotlib & seaborn. Matplotlib is still good in most of the groups though ggplot2 beats it by a small margin in social sciences. Engineering(comp and non-comp) form the majority of visualization libraries users.</p>
    <p><b>Role</b>: Statisticians largely prefer R based packages like ggplot2 and shiny. The users in analyst based roles prefer both R and python based tools equally. Data scientist and analyst based roles are the primary users of visualization libraries.</p>
    <p><b>Industry</b>: Most of the groups prefer python based tools like Matplotlib and seaborn. The users working in Retails, Non-profit, and Insurance also like ggplot2 and shiny. Apart from computer/Technology, students and academicians are the primary users of vizualization libraries.</p>
    <p><b>Salary</b>: The gap between matplotlib and ggplot2 decreases as we move to higher salary levels. The share of D3 users show upward trend if we increase the salary. The lowest salary band and highest salary band users form the major part of vizualization users.</p> 
    <p><b>Experience</b>: As the experience level increases D3 & Shiny show upward trend while Seaborn shows negative trend. Matplotlib beats other packages at every experience level. The total number of responses at a experience level decreases as we move to higher levels of experience.</p>
</div>"""

plot_main('Q21', 'Vizualization Tools', html_string)
html_string = """<div id="Q27">
<p><b>Count</b>: AWS EC2 beats its nearest competitor by a big difference which speaks of its domination in cloud computing markets. The popular methods for cloud computing are virtual servers and event -driven serverless computing and AWS is ahead in both services. Next in line is Google followed by Microsoft's azure with IBM far behind.</p>
<p><b>Age</b>: The share of users who haven't used a cloud computing product is highest in the lowermost age group. One reason for this effect could be the requirement of a valid credit card which is not available to students in countries like India. The AWS EC2 share is lowest in lowermost age groups when compared to other age groups. GCE share is more or less same in all groups. Azure share increases as the age level increases and eclipses GCE in some groups.</p>
<p><b>Country</b>: The share of users who don't use a cloud computing service is very high in Russia and China and lowest in America. Azure has the biggest presence in Russia and the lowest in China. Surprisingly GCE has the lowest share in America when compared to its share in other countries.</p>
<p><b></b></p>
<p><b>Degree Major</b>: It is surprising that the highest share of virtual server users is from non-science branches like Fine arts and Humanities. Also, the humanities group has the lowest percentage of users not using a cloud computing service.</p>
<p><b>Role</b>: We see a lot more users availing serverless computing service in this plot. Salesperson, Statisticians, Data analysts, software engineers, and data journalists roles are the most prominent users of serverless computing. The whole share of serverless computing is divided between AWS and Google with little going to Microsoft's Azure.</p>
<p><b></b></p>
<p><b>Salary</b>: The AWS EC2 has the smallest user base in the lowest salary band when compared to its share in other salary bands. Its share rises as we move to higher salary levels and the same is true for AWS lambda also. GCE share is more or less constant in all groups. Also, the share of users who haven't used a cloud computing service decreases if we increase salary levels.</p>
<p><b></b></p>
</div>"""

plot_main('Q27', 'Cloud computing Products', html_string)
def plot1_28(Q1, Q2, nrows, ncols, name, zbreak=None, index=None):
    
    cols = list(responses_cols_dict[Q1].keys())+[Q2]
    sliced_data = qas.loc[1:,cols]
    
    
    new_cols = [ responses_cols_dict[Q1][key] for key in responses_cols_dict[Q1] ] + [Q2]
    
    sliced_data.columns = new_cols
    
    sliced_df = (sliced_data.groupby(Q2)[list(responses_cols_dict[Q1].values())].count()
                                .sort_index().reindex(index)).drop('None', axis=1)
    
    np_cols = list(sliced_df.sum(axis=0).sort_values(ascending=False)[0:20].index)
    sliced_df = sliced_df[np_cols]
  
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize=(20,10), sharex=True, sharey=True)
    fig.suptitle(name, size='large')
    gp_df = sliced_df.sum(axis=1)
    z = 0
    for i in range(nrows):
        for j in range(ncols):
            
            if zbreak :
                z = i*(nrows - 2) + j
                if (z >= zbreak):
                    break
            else:
                z = i*(nrows - 1) + j
            
            barplt = sns.barplot(x = sliced_df.columns, y = sliced_df.iloc[z]/gp_df.iloc[z], ax=ax[i,j])
            barplt.set_xticklabels( wrap_labels(barplt.get_xticklabels()), rotation=45, ha='right')
            barplt.set_title('Total number of responses in {group}: {num}'.format(group = sliced_df.index[z],
                                                                             num = gp_df.iloc[z]))
    ax[i,j].set_xticklabels( wrap_labels2(list(sliced_df.columns)), rotation=45, ha='right')
    plt.show(fig)



def plot3_28(Q1, Q2, name, index=None):
    cols = list(responses_cols_dict[Q1].keys())+[Q2]
    sliced_data = qas.loc[1:,cols]
    
    new_cols = [ responses_cols_dict[Q1][key] for key in responses_cols_dict[Q1] ] + [Q2]
    sliced_data.columns = new_cols
    sliced_df = (sliced_data.groupby(Q2)[list( responses_cols_dict[Q1].values() )].count().
                                reindex(index).drop('None', axis=1))
    sliced_df_sum = sliced_df.sum(axis=1)
    np_cols = list(sliced_df.sum(axis=0).sort_values(ascending=False)[0:20].index)
    sliced_df = sliced_df[np_cols]
    values = sliced_df.values/(sliced_df_sum.values.reshape(sliced_df_sum.shape[0],1)+1)
    sliced_df = pd.DataFrame(values*100, index=sliced_df.index, columns=sliced_df.columns)

    fig, ax = plt.subplots(figsize=(16,10))
    fig.suptitle(name, size='large')
    ax = sns.heatmap(sliced_df, annot=True, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    bx = ax.twinx()

    sliced_df.index = sliced_df_sum.values.tolist()
    bx = sns.heatmap(sliced_df, annot=True, ax=bx, cbar=False)
    bx.set_yticklabels(sliced_df_sum.values, rotation=360)
    plt.show(fig)


def plot_main_28(Q1, varname, html_string):
    
    display( HTML(html.format(Q=Q1)))
    plot2( Q1, 'Q2', 'Distribution of {var}'.format(var = varname))
    plot1_28( Q1, 'Q2', 3,2, "Distribution of {var} with User's age".format(var = varname),)
    plot1_28( Q1, 'Q3', 3,2, "Distribution of {var} with User's Country ".format(var = varname))
    plot1_28( Q1, 'Q4', 4,2, "Distribution of {var} with User's Education".format(var = varname), zbreak=7,
                             index = education_index)
    plot3_28( Q1, 'Q5', "Distribution of {var} with User's Degree Major".format(var = varname))
    plot3_28( Q1, 'Q6', "Distribution of {var} with User's Role".format(var = varname))
    plot3_28( Q1, 'Q7', "Distribution of {var} with User's Industry".format(var = varname))
    plot1_28( Q1, 'Q9', 3,2, "Distribution of {var} with User's Salary".format(var = varname), 
                             index = salary_index)
    plot3_28( Q1, 'Q8', "Distribution of {var} with User's Experience".format(var = varname), 
                              experience_index)
    display(HTML(html_string))
    display( HTML("<script> showPlot{Q}(0, $('#first{Q}')); </script>".format(Q=Q1)))
    

html_string="""<div id="Q28">
    <p><b>Count</b>: It seems most of the users have not used a machine learning product. The top 4 products are a machine learning as a service product which probably indicates that users prefer to training their own models instead of using a polished api product. SAS(1973) and Rapidminer(2001) are more than a decade old products. It is interesting to note that top ml products are from Google, though it was behind Amazon in cloud computing products as seen in earlier segments. In the middle, we see a lot of IBM's Watson and Azure's AI products. Most of the api products like speech, vision and natural language are not more than two years old and will definitely see a lot of improvement.</p>
    <p><b>Age</b>: Google has the highest penetration in the lowermost age group with a large number of users for its api products. The share of both Azure machine learning studio and Rapidminer rises as we move to higher age groups. Rapidminer has an overwhelmingly large presence among the seniormost age group. The middle age groups form a large part of ml products users.</p>
    <p><b>Country</b>: SAS has the lowest presence among Indian and Brazilians, though it's quite popular among Americans. Rapidminer has the lowest presence among Chinese when compared to its share in other countries. Google's services and products from Other vendors are quite popular among Chinese. The users from Other countries are primary users of ml products.</p>
    <p><b>Education</b>: The high-school group prefers ml products which have been launched recently within 3-4 years. The Ph.D. holders prefer SAS and Cloudera more than any other ml products.</p>
    <p><b>Degree Major</b>: All the branches related to statistics like maths & stats, social sciences prefer SAS more than any other ml products.</p>
    <p><b>Role</b>: The users in analyst and statistician roles heavily prefer SAS. The data journalists mostly use ml engines of Google and Azure.</p>
    <p></p>
    <p><b>Salary</b>: On average higher incomer group prefer SAS more than lower income groups. The share of Sagemaker users rises as we move to higher levels of income.</p>
    <p></p>
</div>"""
plot_main_28('Q28', 'ML Products', html_string)
html_string = """<div id="Q29">
    <p><b>Count</b>: We can see two groups in the plot traditional databases and relatively new colud based database products. Each database management system in the top five is traditional and more than a decade old. All the cloud hosted relational database are far behind the traditional databases. MySQL is the leader & well ahead of its nearest competitors. It is followed by PostgreSQL, SQLite, Microsoft SQL server, Oracle Database, and Microsoft Access. It is important to note that the number of users who haven't use a database is greater than every cloud database product. Among the cloud databases Amazon is ahead with its relational database(2009) and dynamoDB(2012) service. DynamoDB was launched after both Azure Sql(2010) & Cloud Sql(2011). IBM cloud database products are have captured little user base. </p> 
    <p><b>Age</b>: MySQL is the first choice for every age group. The youngest and seniormost age group give the second preference to SQLite. The share of cloud database products is low among all age groups. 
</p>
    <p><b>Country</b>: MySQL is the most preferred database product in all countries especially China. Indians and Chinese give low preference to PostgreSQL when compared to other countries.</p>
    <p><b>Education</b>: The high-school group has shown interest in cloud database products like dynamoDB and azure sql. As the age level increases, we see a downward trend among cloud database users. MySQL is still the clear favorite. </p>
    <p><b>Degree Major</b>: The users having a computer science related major degree form disproportionately large share of database products users. The non-comp users have shown a bit of interest in cloud database products. MySQL is the leader and next three positions are dividend among PostgreSQL, SQLite, and MS SQL server.</p>
    <p><b>Role</b>: The database engineer prefers Microsoft SQL Server more than MySQL. The users with analyst roles have a strong liking for commercial traditional database products like Microsoft sql server and Oracle database. </p>
    <p><b></b>: </p>
    <p><b>Salary</b>: The MySQL's share decreases as we move towards higher salary levels. We see a small interest in cloud database products from high-salary levels. The big margin between MySQL and other database products as seen in the lowest salary level decreases if the salary increases. </p>
    <p><b>Experience</b>: There is a downward trend in MySQL and PostgreSQL users as the experience of the user increases while the opposite is true for Microsoft access. MySQL is again the most preferred database.</p>
</div>"""

plot_main_28('Q29', 'Relational Database', html_string)
html_string="""<div id="Q30">
<p><b>Count</b>: Most of the users haven't used a commercial big data analytics product. Google's Bigquery(2011) is the most popular product followed by amazon's redshift(2012) and Databricks(2013). In big data analytics segment AWS is clearly ahead of both Google and aws while IBM is far behind.</p>
<p><b>Age</b>: Lowermost age group users largely google product specially Bigquery. AWS redshift has the highest share in the age group 25-34 and decreases as we move to higher age levels. The percentage of users preferring Databricks platform keeps on rising as we towards higher age levels.</p>
<p><b>Country</b>: Of all the countries Chinese users have the least preference for redshift & Teradata while they prefer Bigquery, Dataflow, and Other product probably Alibaba. Google Bigquery is quite popular in Russia and the same can be said true for AWS redshift in America. </p>
<p><b>Education</b>: Google's Bigquery is most preferred big data product in lowermost education group and its share is far less in other groups. The middle age groups again form the majority of the users. Databricks is popular in higher education groups.</p>
<p></p>
<p><b>Role</b>: The users with developer advocate & data engineer roles heavily prefer Microsoft analysis service while google's pub/sub & AWS batch are favorites among data journalists. Big query is a big favorite among most of the groups.</p>
<p></p>
<p><b>Salary</b>: As the salary level rises the share of redshift and Databricks increases while google's pub/sub share decreases.</p>
<p><b>Experience</b>: On an average lesser number of experienced users are using Bigquery when compared to the share of low experienced groups</p>
</div>
"""

plot_main_28('Q30', 'Big Data Products', html_string)