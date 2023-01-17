# Generic Libraries

import numpy as np

import pandas as pd



# Visualisation Libraries

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

import warnings

from matplotlib import cm



pd.plotting.register_matplotlib_converters()

%matplotlib inline

plt.style.use('seaborn-whitegrid')

pd.set_option('display.max_columns', 500)

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.2f}'.format
url = '../input/churn-model/Churn_Modelling.csv'

data = pd.read_csv(url, header='infer')
data.shape
data.dtypes
#Dropping the Row Number & Customer Id columns

data = data.drop(columns=['RowNumber','CustomerId'], axis=1)
# Changing the data types for certain columns to 'Category'



cols = set(data.columns)

cols_numeric = set(['CreditScore','Age', 'Tenure','Balance','NumOfProducts','EstimatedSalary'])

cols_obj = set(['Surname','Geography'])

cols_category = list(cols - cols_numeric - cols_obj)



for x in cols_category:

    data[x] = data[x].astype('category')
data.describe().transpose()
# Let's construct a function that shows the summary and density distribution of a numerical columns



def summary(x):

    x_min = data[x].min()

    x_max = data[x].max()

    Q1 = data[x].quantile(0.25)

    Q2 = data[x].quantile(0.50)

    Q3 = data[x].quantile(0.75)

    x_mean = data[x].mean()

    print(f'6 Point Summary of {x.capitalize()} Attribute:\n'

          f'{x.capitalize()}(min)   : {x_min}\n'

          f'Q1                      : {Q1}\n'

          f'Q2(Median)              : {Q2}\n'

          f'Q3                      : {Q3}\n'

          f'{x.capitalize()}(max)   : {x_max}\n'

          f'{x.capitalize()}(mean)  : {round(x_mean)}')



    fig = plt.figure(figsize=(15, 10))

    plt.subplots_adjust(hspace = 0.6)

    sns.set_palette('pastel')

    

    plt.subplot(221)

    ax1 = sns.distplot(data[x], color = 'r')

    plt.title(f'{x.capitalize()} Density Distribution')

    

    plt.subplot(222)

    ax2 = sns.violinplot(x = data[x], palette = 'Accent', split = True)

    plt.title(f'{x.capitalize()} Violinplot')

    

    plt.subplot(223)

    ax2 = sns.boxplot(x=data[x], palette = 'cool', width=0.7, linewidth=0.6)

    plt.title(f'{x.capitalize()} Boxplot')

    

    plt.subplot(224)

    ax3 = sns.kdeplot(data[x], cumulative=True)

    plt.title(f'{x.capitalize()} Cumulative Density Distribution')

    

    plt.show()
#Summary Age

summary('Age')
#Summary EstimatedSalary

summary('EstimatedSalary')
# Create a function that returns a Pie chart for the categorical variables:

def cat_view(country):

    """

    Function to create a Bar chart and a Pie chart for categorical variables.

    """

    from matplotlib import cm

    color1 = cm.inferno(np.linspace(.4, .8, 30))

    color2 = cm.viridis(np.linspace(.4, .8, 30))

    

    fig, ax = plt.subplots(2, 2, figsize=(15, 7))

    explode = (0.1, 0)

    

    #Creating a country dataframe

    country_df = data[data['Geography'] == country]



    """

    Draw Gender Pie Chart on first subplot.

    """    

    gndr = country_df.groupby('Gender').size()



    gndr_mydata_values = gndr.values.tolist()

    gndr_mydata_index = gndr.index.tolist()



    def func(pct, allvals):

        absolute = int(pct/100.*np.sum(allvals))

        return "{:.1f}%\n({:d})".format(pct, absolute)



    wedges, texts, autotexts = ax[0,0].pie(gndr_mydata_values, autopct=lambda pct: func(pct, gndr_mydata_values), textprops=dict(color="w"), explode=explode)

    ax[0,0].legend(wedges, gndr_mydata_index,title="Index", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=12, weight="bold")

    ax[0,0].set_title(f'{country.capitalize()} Gender Distribution')

    

    

    """

    Draw Has Credit Card Pie Chart on second subplot.

    """    

    cc = country_df.groupby('HasCrCard').size()



    cc_mydata_values = cc.values.tolist()

    #cc_mydata_index = cc.index.tolist()

    cc_mydata_index = ['no','yes']



    def func(pct, allvals):

        absolute = int(pct/100.*np.sum(allvals))

        return "{:.1f}%\n({:d})".format(pct, absolute)



    wedges, texts, autotexts = ax[0,1].pie(cc_mydata_values, autopct=lambda pct: func(pct, cc_mydata_values), textprops=dict(color="w"),explode=explode)

    ax[0,1].legend(wedges, cc_mydata_index,title="Index", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=12, weight="bold")

    ax[0,1].set_title(f'{country.capitalize()} Credit Card Ownership Distribution')    



    

    """

    Draw Is Active Member Pie Chart on third subplot.

    """    

    am = country_df.groupby('IsActiveMember').size()



    am_mydata_values = am.values.tolist()

    #am_mydata_index = am.index.tolist()

    am_mydata_index = ['no','yes']



    def func(pct, allvals):

        absolute = int(pct/100.*np.sum(allvals))

        return "{:.1f}%\n({:d})".format(pct, absolute)



    wedges, texts, autotexts = ax[1,0].pie(am_mydata_values, autopct=lambda pct: func(pct, am_mydata_values), textprops=dict(color="w"),explode=explode)

    ax[1,0].legend(wedges, am_mydata_index,title="Index", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=12, weight="bold")

    ax[1,0].set_title(f'{country.capitalize()} Active Membership Distribution')    

    



    """

    Draw Member Exited Pie Chart on fourth subplot.

    """    

    ex = country_df.groupby('Exited').size()



    ex_mydata_values = ex.values.tolist()

    #ex_mydata_index = ex.index.tolist()

    ex_mydata_index = ['no','yes']

    



    def func(pct, allvals):

        absolute = int(pct/100.*np.sum(allvals))

        return "{:.1f}%\n({:d})".format(pct, absolute)



    wedges, texts, autotexts = ax[1,1].pie(ex_mydata_values, autopct=lambda pct: func(pct, ex_mydata_values), textprops=dict(color="w"),explode=explode)

    ax[1,1].legend(wedges, ex_mydata_index,title="Index", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=12, weight="bold")

    ax[1,1].set_title(f'{country.capitalize()} Member Exited Distribution')    

    

    



    fig.tight_layout()

    plt.show()
#Category Column Analysis - France

cat_view('France')
#Category Column Analysis - Germany

cat_view('Germany')
#Category Column Analysis - Spain

cat_view('Spain')
# Create a function to categorize the credit score

def CreditScore_Cat(score):

    if 300 <= score <= 629:

        return 'bad'

    elif 630 <= score <= 689:

        return 'fair'

    elif 690 <= score <= 719:

        return 'good'

    else:

        return 'excellent'



data['CreditScore_Cat'] = data['CreditScore'].apply(CreditScore_Cat)

def creditScore_Viz():

    """

    Function to create Pie chart for Credit Score categorical variables.

    """

    plt.figure(figsize=(14, 9))

    explode = (0.1,0.1,0.1,0.1)

    colors = ['#ff9999','#66b3ff','#618739','#ffcc99']

    

    # Create 2x2 sub plots

    gs = gridspec.GridSpec(2, 2)

    

    #Creating a country dataframe

    cs_fr_df = data[data['Geography'] == 'France']

    cs_gr_df = data[data['Geography'] == 'Germany']

    cs_sp_df = data[data['Geography'] == 'Spain']

    

    

    """

    Draw France Pie Chart on first subplot.

    """    

    

    ax = plt.subplot(gs[0, 0]) # row 0, col 0



    

    cs_fr = cs_fr_df.groupby('CreditScore_Cat').size()



    cs_fr_mydata_values = cs_fr.values.tolist()

    cs_fr_mydata_index = cs_fr.index.tolist()



    def func(pct, allvals):

        absolute = int(pct/100.*np.sum(allvals))

        return "{:.1f}%\n({:d})".format(pct, absolute)



    wedges, texts, autotexts = ax.pie(cs_fr_mydata_values, autopct=lambda pct: func(pct, cs_fr_mydata_values), textprops=dict(color="w"),explode=explode, 

                                      colors=colors,shadow=True)

    ax.legend(wedges, cs_fr_mydata_index,title="Index", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=12, weight="bold")

    ax.set_title('France Credit Score Distribution', fontsize=14)

    

    

    """

    Draw Germany Pie Chart on first subplot.

    """    

    ax = plt.subplot(gs[0, 1]) # row 0, col 0

    

    cs_gr = cs_gr_df.groupby('CreditScore_Cat').size()



    cs_gr_mydata_values = cs_gr.values.tolist()

    cs_gr_mydata_index = cs_gr.index.tolist()



    def func(pct, allvals):

        absolute = int(pct/100.*np.sum(allvals))

        return "{:.1f}%\n({:d})".format(pct, absolute)



    wedges, texts, autotexts = ax.pie(cs_gr_mydata_values, autopct=lambda pct: func(pct, cs_gr_mydata_values), textprops=dict(color="w"),explode=explode, 

                                      colors=colors,shadow=True)

    ax.legend(wedges, cs_gr_mydata_index,title="Index", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=12, weight="bold")

    ax.set_title('Germany Credit Score Distribution', fontsize=14)  

        

    

    """

    Draw Spain Pie Chart on first subplot.

    """    

    ax = plt.subplot(gs[1, :])

    

    cs_sp = cs_sp_df.groupby('CreditScore_Cat').size()



    cs_sp_mydata_values = cs_sp.values.tolist()

    cs_sp_mydata_index = cs_sp.index.tolist()



    def func(pct, allvals):

        absolute = int(pct/100.*np.sum(allvals))

        return "{:.1f}%\n({:d})".format(pct, absolute)



    wedges, texts, autotexts = ax.pie(cs_sp_mydata_values, autopct=lambda pct: func(pct, cs_sp_mydata_values), textprops=dict(color="w"),explode=explode, 

                                      colors=colors,shadow=True)

    ax.legend(wedges, cs_sp_mydata_index,title="Index", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=12, weight="bold")

    ax.set_title('Spain Credit Score Distribution', fontsize=14)       

       

    plt.tight_layout()

    plt.show()
#Credit Score Category Visualisation Per Country

creditScore_Viz()
#Taking a Backup

data_backup = data.copy()
#add a new Age Group Column

age_cat = pd.cut(data.Age,bins=[10,31,36,44,60,95],labels=['18-31_Grp','32-36_Grp','37-44_Grp','45-60_Grp','61-95_Grp'])

data.insert(13,'Age_Groups',age_cat)
# Create a function that returns a Bar char for the categorical variables:

def BivAnalysis1(country):

    """

    Function to create a Bar chart for numerical variables.

    """

    color1 = cm.inferno(np.linspace(.4, .8, 30))

    color2 = cm.viridis(np.linspace(.4, .8, 30))

    

    fig, ax = plt.subplots(2, 2, figsize=(20, 12))

        

    #Creating a country dataframe

    country_df = data[data['Geography'] == country]



    """

    Draw Gender vs Age Group vs Tenure Bar Chart on first subplot.

    """    

    df1 = pd.pivot_table(country_df, index = ['Age_Groups'], columns = ['Gender'], values = ['Tenure'], aggfunc = np.mean)

    

    labels1 = df1.index.tolist()

    female1 = df1.values[:, 0].tolist()

    male1 = df1.values[:, 1].tolist()

    

    l1 = np.arange(len(labels1))  # the label locations

    width = 0.35  # the width of the bars

    

    rects11 = ax[0,0].bar(l1 - width/2, female1, width, label='Female', color = color1)

    rects12 = ax[0,0].bar(l1 + width/2, male1, width, label='Male', color = color2)



    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax[0,0].set_ylabel('Tenure')

    ax[0,0].set_title('Age Groups v/s Avg. Tenure Bar Graph')

    ax[0,0].set_xticks(l1)

    ax[0,0].set_xticklabels(labels1)

    ax[0,0].legend()

    

    def autolabel(rects):

        for rect in rects:

            height = rect.get_height()

            ax[0,0].annotate('{:.1f}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext=(0, -50), textcoords="offset points",fontsize = 'small',

                           ha='center', va='bottom',color ='white')



    autolabel(rects11)

    autolabel(rects12)

    

    

    """

    Draw Gender vs Age Group vs Num. of Products Bar Chart on second subplot.

    """    

    df2 = pd.pivot_table(country_df, index = ['Age_Groups'], columns = ['Gender'], values = ['NumOfProducts'], aggfunc = np.mean)

    

    labels2 = df2.index.tolist()

    female2 = df2.values[:, 0].tolist()

    male2 = df2.values[:, 1].tolist()

    

    l2 = np.arange(len(labels2))  # the label locations

    width = 0.35  # the width of the bars

    

    rects21 = ax[0,1].bar(l2 - width/2, female2, width, label='Female', color = color1)

    rects22 = ax[0,1].bar(l2 + width/2, male2, width, label='Male', color = color2)



    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax[0,1].set_ylabel('Number of Products')

    ax[0,1].set_title('Age Groups v/s Avg. Num of Products Bar Graph')

    ax[0,1].set_xticks(l2)

    ax[0,1].set_xticklabels(labels2)

    ax[0,1].legend()

    

    def autolabel(rects):

        for rect in rects:

            height = rect.get_height()

            ax[0,1].annotate('{:.1f}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext=(0, -50), textcoords="offset points",fontsize = 'small',

                           ha='center', va='bottom',color ='white')



    autolabel(rects21)

    autolabel(rects22)   



    

    """

    Draw Gender vs Age Group vs Balance Bar Chart on third subplot.

    """    

    df3 = pd.pivot_table(country_df, index = ['Age_Groups'], columns = ['Gender'], values = ['Balance'], aggfunc = np.mean)

    

    labels3 = df3.index.tolist()

    female3 = df3.values[:, 0].tolist()

    male3 = df3.values[:, 1].tolist()

    

    l3 = np.arange(len(labels3))  # the label locations

    width = 0.35  # the width of the bars

    

    rects31 = ax[1,0].bar(l3 - width/2, female3, width, label='Female', color = color1)

    rects32 = ax[1,0].bar(l3 + width/2, male3, width, label='Male', color = color2)



    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax[1,0].set_ylabel('Balance')

    ax[1,0].set_title('Age Groups v/s Avg. Balance Bar Graph')

    ax[1,0].set_xticks(l3)

    ax[1,0].set_xticklabels(labels3)

    ax[1,0].legend()

    

    def autolabel(rects):

        for rect in rects:

            height = rect.get_height()

            ax[1,0].annotate('{:.1f}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext=(0, -50), textcoords="offset points",fontsize = 'small',

                           ha='center', va='bottom',color ='white')



    autolabel(rects31)

    autolabel(rects32)  

    

    



    """

    Draw Gender vs Age Group vs Estimated Salary Bar Chart on fourth subplot.

    """    

    df4 = pd.pivot_table(country_df, index = ['Age_Groups'], columns = ['Gender'], values = ['EstimatedSalary'], aggfunc = np.mean)

    

    labels4 = df4.index.tolist()

    female4 = df4.values[:, 0].tolist()

    male4 = df4.values[:, 1].tolist()

    

    l4 = np.arange(len(labels4))  # the label locations

    width = 0.35  # the width of the bars

    

    rects41 = ax[1,1].bar(l4 - width/2, female4, width, label='Female', color = color1)

    rects42 = ax[1,1].bar(l4 + width/2, male4, width, label='Male', color = color2)



    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax[1,1].set_ylabel('Estimated Salary')

    ax[1,1].set_title('Age Groups v/s Avg. Estimated Salary Bar Graph')

    ax[1,1].set_xticks(l4)

    ax[1,1].set_xticklabels(labels4)

    ax[1,1].legend()

    

    def autolabel(rects):

        for rect in rects:

            height = rect.get_height()

            ax[1,1].annotate('{:.1f}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext=(0, -50), textcoords="offset pixels",fontsize = 'small',

                           ha='center', va='bottom', color ='white')



    autolabel(rects41)

    autolabel(rects42)    

    

    fig.tight_layout(pad = 2.5)

    plt.show()
#Analysis of French Customers

BivAnalysis1('France')
#Analysis of German Customers

BivAnalysis1('Germany')
#Analysis of Spanish Customers

BivAnalysis1('Spain')
# Create a function that returns a Bar char for the categorical variables:

def BivAnalysis2(country):

    """

    Function to create a Bar chart for numerical variables.

    """

    color1 = cm.inferno(np.linspace(.4, .8, 30))

    color2 = cm.viridis(np.linspace(.4, .8, 30))

    

    fig, ax = plt.subplots(2, 2, figsize=(20, 12))

        

    #Creating a country dataframe

    country_df = data[data['Geography'] == country]



    """

    Draw Gender vs Credit Score vs Tenure Bar Chart on first subplot.

    """    

    df1 = pd.pivot_table(country_df, index = ['CreditScore_Cat'], columns = ['Gender'], values = ['Tenure'], aggfunc = np.mean)

    

    labels1 = df1.index.tolist()

    female1 = df1.values[:, 0].tolist()

    male1 = df1.values[:, 1].tolist()

    

    l1 = np.arange(len(labels1))  # the label locations

    width = 0.35  # the width of the bars

    

    rects11 = ax[0,0].bar(l1 - width/2, female1, width, label='Female', color = color1)

    rects12 = ax[0,0].bar(l1 + width/2, male1, width, label='Male', color = color2)



    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax[0,0].set_ylabel('Tenure')

    ax[0,0].set_title('Credit Scores  v/s Avg. Tenure Bar Graph')

    ax[0,0].set_xticks(l1)

    ax[0,0].set_xticklabels(labels1)

    ax[0,0].legend()

    

    def autolabel(rects):

        for rect in rects:

            height = rect.get_height()

            ax[0,0].annotate('{:.1f}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext=(0, -50), textcoords="offset points",fontsize = 'small',

                           ha='center', va='bottom',color ='white')



    autolabel(rects11)

    autolabel(rects12)

    

    

    """

    Draw Gender vs Credit Score vs Num. of Products Bar Chart on second subplot.

    """    

    df2 = pd.pivot_table(country_df, index = ['CreditScore_Cat'], columns = ['Gender'], values = ['NumOfProducts'], aggfunc = np.mean)

    

    labels2 = df2.index.tolist()

    female2 = df2.values[:, 0].tolist()

    male2 = df2.values[:, 1].tolist()

    

    l2 = np.arange(len(labels2))  # the label locations

    width = 0.35  # the width of the bars

    

    rects21 = ax[0,1].bar(l2 - width/2, female2, width, label='Female', color = color1)

    rects22 = ax[0,1].bar(l2 + width/2, male2, width, label='Male', color = color2)



    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax[0,1].set_ylabel('Number of Products')

    ax[0,1].set_title('Credit Score v/s Avg. Num of Products Bar Graph')

    ax[0,1].set_xticks(l2)

    ax[0,1].set_xticklabels(labels2)

    ax[0,1].legend()

    

    def autolabel(rects):

        for rect in rects:

            height = rect.get_height()

            ax[0,1].annotate('{:.1f}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext=(0, -50), textcoords="offset points",fontsize = 'small',

                           ha='center', va='bottom',color ='white')



    autolabel(rects21)

    autolabel(rects22)   



    

    """

    Draw Gender vs Credit Score vs Balance Bar Chart on third subplot.

    """    

    df3 = pd.pivot_table(country_df, index = ['CreditScore_Cat'], columns = ['Gender'], values = ['Balance'], aggfunc = np.mean)

    

    labels3 = df3.index.tolist()

    female3 = df3.values[:, 0].tolist()

    male3 = df3.values[:, 1].tolist()

    

    l3 = np.arange(len(labels3))  # the label locations

    width = 0.35  # the width of the bars

    

    rects31 = ax[1,0].bar(l3 - width/2, female3, width, label='Female', color = color1)

    rects32 = ax[1,0].bar(l3 + width/2, male3, width, label='Male', color = color2)



    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax[1,0].set_ylabel('Balance')

    ax[1,0].set_title('Credit Score v/s Avg. Balance Bar Graph')

    ax[1,0].set_xticks(l3)

    ax[1,0].set_xticklabels(labels3)

    ax[1,0].legend()

    

    def autolabel(rects):

        for rect in rects:

            height = rect.get_height()

            ax[1,0].annotate('{:.1f}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext=(0, -50), textcoords="offset points",fontsize = 'small',

                           ha='center', va='bottom',color ='white')



    autolabel(rects31)

    autolabel(rects32)  

    

    



    """

    Draw Gender vs Credit Score vs Estimated Salary Bar Chart on fourth subplot.

    """    

    df4 = pd.pivot_table(country_df, index = ['CreditScore_Cat'], columns = ['Gender'], values = ['EstimatedSalary'], aggfunc = np.mean)

    

    labels4 = df4.index.tolist()

    female4 = df4.values[:, 0].tolist()

    male4 = df4.values[:, 1].tolist()

    

    l4 = np.arange(len(labels4))  # the label locations

    width = 0.35  # the width of the bars

    

    rects41 = ax[1,1].bar(l4 - width/2, female4, width, label='Female', color = color1)

    rects42 = ax[1,1].bar(l4 + width/2, male4, width, label='Male', color = color2)



    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax[1,1].set_ylabel('Estimated Salary')

    ax[1,1].set_title('Credit Score v/s Avg. Estimated Salary Bar Graph')

    ax[1,1].set_xticks(l4)

    ax[1,1].set_xticklabels(labels4)

    ax[1,1].legend()

    

    def autolabel(rects):

        for rect in rects:

            height = rect.get_height()

            ax[1,1].annotate('{:.1f}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext=(0, -50), textcoords="offset pixels",fontsize = 'small',

                           ha='center', va='bottom', color ='white')



    autolabel(rects41)

    autolabel(rects42)    

    

    fig.tight_layout(pad = 2.5)

    plt.show()
#Analysis of French Customers

BivAnalysis2('France')
#Analysis of French Customers

BivAnalysis2('Germany')
#Analysis of French Customers

BivAnalysis2('Spain')