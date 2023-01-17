# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import pi

from scipy import stats 

from sklearn.decomposition import PCA

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.cluster import KMeans, DBSCAN

from sklearn.metrics import mean_squared_error



import warnings











#########################################################

#Plotting tools!

#########################################################

from bokeh.plotting import figure

from bokeh.io import push_notebook, show, show, output_notebook

from bokeh.models import (

    ColumnDataSource,

    HoverTool,

    LinearColorMapper,

    BasicTicker,

    PrintfTickFormatter,

    ColorBar,

    FactorRange,

    

)

from bokeh.models import ContinuousColorMapper

from bokeh.models.glyphs import VBar

from bokeh.core.properties import ColorSpec, value

from bokeh.layouts import gridplot

from bokeh.palettes import Viridis256





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

output_notebook()

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv("../input/HR_comma_sep.csv", index_col=False)

factor_scanned_data = pd.DataFrame()
print(data.head(5))
for i in data.columns.values:

    print(i + " " + str(data[i].unique()))
data.describe()
def code_as_factor(factors, factorName):

    '''

    Codes a list of factors to a numerical key value for each factor

    '''

    coded_factors = []

    codes = factors.unique().tolist()

    for i in factors:

        coded_factors.append(codes.index(i))

    return pd.DataFrame({factorName : coded_factors})



def code_categorical_list(data, categoricals):

    '''

    Given a list of categorical variables the list of variables will be coded to a set of 

    '''

    for i in categoricals:

        tmp = data[i]

        data = data.drop(labels=i, axis=1)

        data[i] = code_as_factor(tmp, i)

    return data



def factor_test(column):

    '''

    boolean test if factor variable

    '''

    return type(column.iloc[0]) == str

def build_histos(data):

    hists = []

    f_s_d = data.copy()

    for i in data.columns.values:

        #print(i)

        data_for_hists = data[i]

        factor = False

        if factor_test(data_for_hists):

            data_for_hists = code_as_factor(data_for_hists, i)

            factor = True

        #output_file("./histograms/"+i+".html")

        

        hist, edges = np.histogram(data_for_hists, bins=50, density=True)

        if factor:

            p1 = figure(title="Histogram of "+ i,tools="save", background_fill_color="#E8DDCB", x_range=data[i].unique().tolist(), plot_height=400)

            p1.xaxis.major_label_orientation = pi / 3

            

        else:

            p1 = figure(title="Histogram of "+ i,tools="save", background_fill_color="#E8DDCB", plot_height=400)

        p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color=None )

        #export_png(p1, filename="./histograms/"+i+".png")

        hists.append(p1)

        f_s_d[i] = data_for_hists

    push_notebook(handle=show(gridplot(hists, ncols=2, plot_width=400,plot_height=400, toolbar_location=None), notebook_handle=True))

    return f_s_d

factor_scanned_data = build_histos(data.copy())



#Box + Whisker

def whisker(data, target, variables):

    boxes = []

    for variable in variables:

        cats = data[target].unique().tolist()

        dat = data.groupby(target)[variable]

        groups = dat

        q1 = groups.quantile(q=0.25)

        q2 = groups.quantile(q=0.5)

        q3 = groups.quantile(q=0.75)

        

        #outlier detections

        iqr = q3 - q1

        upper = q3 + 1.5*iqr

        lower = q1 - 1.5*iqr

        

        maxy = groups.max()

        miny = groups.min()

        

        def outliers(group):

            cat = group.name

            return group[(group > upper.loc[cat]) | (group < lower.loc[cat])]

        out = groups.apply(outliers).dropna()



        if not out.empty:

            outx = []

            outy = []

            for cat in cats:

                # only add outliers if they exist

                if not out.loc[cat].empty:

                    for value in out[cat]:

                        outx.append(cat)

                        outy.append(value)



        p = figure(background_fill_color="#EFE8E2", title="", plot_width=900, plot_height=500, tools="hover", x_axis_label="Condition of "+target, y_axis_label="Value of "+variable)



        # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums

        qmin = groups.quantile(q=0.00)

        qmax = groups.quantile(q=1.00)

        upper.score = qmin

        lower.score = qmax



        # stems

        p.segment(cats, maxy, cats, q3, line_color="black")

        p.segment(cats, miny, cats, q1, line_color="black")



        # boxes

        p.vbar(cats, 0.7, q2, q3, fill_color="#E08E79", line_color="black")

        p.vbar(cats, 0.7, q1, q2, fill_color="#3B8686", line_color="black")



        # whiskers (almost-0 height rects simpler than segments)

        p.rect(cats, miny, 0.2, 0.01, line_color="black")

        p.rect(cats, maxy, 0.2, 0.01, line_color="black")



        # outliers

        if not out.empty:

            p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)



        p.xgrid.grid_line_color = None

        p.ygrid.grid_line_color = "white"

        p.grid.grid_line_width = 2

        p.xaxis.major_label_text_font_size="12pt"

        

        

        boxes.append(p)



    push_notebook(handle=show(gridplot(boxes, ncols=2, plot_width=400, plot_height=400, toolbar_location=None),notebook_handle=True))

    

whisker(data.copy(), 'left', ['satisfaction_level', 'last_evaluation','average_montly_hours' ])

def generate_bar_comparisons(data, target, categories):

    bars=[]

    for category in categories:

        data_grouped = data.groupby([target])[category]

        counts = dict()

        for i in data_grouped.groups.keys():

            counts[i] = (data_grouped.get_group(i).value_counts())

        

        palette = ["#e84d60", "#718dbf"]

        cats = data[category].unique().tolist()

        

        p = figure(x_range=FactorRange(*cats), title=category+" between states of " + target,

           toolbar_location=None, tools="hover", plot_height=500)

        

        cat_counter = 0

        

        for i in counts.keys():

            tmp_frame = counts[i].to_frame()

            tmp_frame[target] = i

            with warnings.catch_warnings():

                warnings.filterwarnings("ignore",category=DeprecationWarning)

                p.vbar(x=tmp_frame.index.tolist(),

                       top=tmp_frame[category], 

                       width=0.5, line_color="white", 

                       fill_color=palette[cat_counter],

                      source=tmp_frame)

            

            cat_counter+=1

            p.xaxis.major_label_orientation = (pi / 3)*-1

        p.select_one(HoverTool).tooltips = [

            (category + ' count', '@'+category),

            (target , '@'+target)

        ]

        

        bars.append(p)

    push_notebook(handle=show(gridplot(bars, ncols=2, plot_width=400,plot_height=400,toolbar_location=None), notebook_handle=True))

    

generate_bar_comparisons(data.copy(), 'left', ['number_project', 'time_spend_company', 'salary', 'sales', 'Work_accident'])  
def remove_factors(data):

    for i in data.columns.values:

        if factor_test(data[i]):

            del data[i]

    return data



def remove_cats(data, categories):

    for i in data.columns.values:

        if i in categories:

            del data[i]

    return data

def heater(data):

    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]

    data = remove_factors(data)

    #data = remove_cats(data, ["Work_accident", "promotion_last_5years"])

    

    correlation = data.corr()

    mapper = LinearColorMapper(palette=colors, low=-1, high=1)

    column_names = data.columns.values.tolist()

    

    p = figure(title="Correlations Between HR Numeric & Binary Variables ", x_range= column_names, y_range=column_names, 

        toolbar_location='below', x_axis_location="above", plot_width=900, plot_height=500,

        tools="hover, pan")

    

    p.grid.grid_line_color = None

    p.axis.axis_line_color = None

    p.axis.major_tick_line_color = None

    p.xaxis.major_label_orientation = pi / 3



    

    for i in range(0, len(column_names)):

        for h in range(0, len(column_names)):

            #this is pretty wasteful but with the data I am working with I have not yet found a better way?

            df_t = pd.DataFrame(data=[correlation.get_value(data.columns.values.tolist()[i], data.columns.values.tolist()[h])], columns=['f'])

            p.rect(x=i+1, y=h+1, source=df_t, width=1, height=1, 

                fill_color={'field': 'f', 'transform': mapper }

                )

            

            p.text(x=i+.75, y=h+.75, text=['{:03.2f}'.format(correlation.get_value(data.columns.values.tolist()[i], data.columns.values.tolist()[h]))])

    

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",

                     ticker=BasicTicker(desired_num_ticks=len(colors)),

                     formatter=PrintfTickFormatter(format="%f"),

                     label_standoff=6, border_line_color=None, location=(0, 0))

    p.add_layout(color_bar, 'right')

    

    p.select_one(HoverTool).tooltips = [

        ('R', '@f'),

    ]

    #output_file("./heatmaps/correlates.html")

    #export_png(p, filename="./heatmaps/correlates.png")

    push_notebook(handle=show(p, notebook_handle=True))

    

def scatter_all(data):

    scatters = []

    for i in data.columns:

        for h in data.columns:

            p = figure(title="Scatter between " + i + " v " + h,

               toolbar_location=None, tools="hover", plot_height=500, x_axis_label= i, y_axis_label=h)

            p.circle(data[i], data[h],fill_color="#75968f", size=10, line_color="white")

            scatters.append(p)

    push_notebook(handle=show(gridplot(scatters, ncols=2, plot_width=400,plot_height=400,toolbar_location=None), notebook_handle=True))

heater(data.copy())

scatter_all(data.copy())
def t_test_all(data, target, continuous_variables):

    target_types = data[target].unique()

    for i in continuous_variables:

        cat_0 = data[data[target]==target_types[0]]

        cat_1 = data[data[target]==target_types[1]]

        print(str(stats.ttest_ind(cat_1[i], cat_0[i])) + "T Test Results for : " + i)



t_test_all(data.copy(), "left", ["satisfaction_level", "average_montly_hours", "number_project", "time_spend_company"])
def chi_sq_all(data, target, categoricals):

    for i in categoricals:

        groupsizes = data.groupby([i, target]).size()

        ctsum = groupsizes.unstack(i)

        # fillna(0) is necessary to remove any NAs which will cause exceptions

        chi2 , p , dof, ex= stats.chi2_contingency(ctsum.fillna(0))

        print("Chi Sq: Test Statistic -- " + str(chi2) +" P Value -- "+ str(p) +" Variable -- " + i)

chi_sq_all(data.copy(), "left", ["salary", "sales", "Work_accident"])
from sklearn.ensemble import ExtraTreesClassifier

#got much of this from the sklearn site

def show_best_predictors(data, target, categoricals):

    data = code_categorical_list(data, categoricals)

    # Build a forest and compute the feature importances

    forest = ExtraTreesClassifier(n_estimators=250,

                                  random_state=0)

    y = data[target].values

    cols = [i for i in list(data.columns) if i != target]

    X= data.ix[:, cols].as_matrix()



    forest.fit(X, y)

    importances = forest.feature_importances_

    #Sort the column names by their importance in the tree model from most-least

    cols_by_imp  = [x for _,x in sorted(zip(importances,cols), reverse=True)]

    dat = pd.DataFrame({'rankings': sorted(importances.tolist(), reverse=True), 'features': cols_by_imp})

    #data vis

    p = figure(x_range=cols_by_imp, title="Importance of variable to tree model",

           toolbar_location=None, tools="hover", plot_height=500)

    p.vbar(x=dat['features'],top=dat['rankings'],

           fill_color="#75968f",width=0.5, line_color="white")

    p.xaxis.major_label_orientation = (pi / 3)*-1

    push_notebook(handle=show(p, notebook_handle=True))

    

show_best_predictors(data.copy(), 'left', ['sales', 'salary', 'Work_accident'])
data_left = factor_scanned_data.copy()[factor_scanned_data['left']==1]

len(data_left['left'])

kmns_data = data_left.copy().drop(['left'], axis=1)

listy_data = kmns_data.values.tolist()





kmeans_errors = []

for i in range(1, 20):

    kmns = KMeans(n_clusters=i).fit(kmns_data)

    centroids = kmns.cluster_centers_

    labels = kmns.labels_ 

    

    true = []

    pred = []

    for i in range(0,len(labels)):

        true.append(listy_data[i])

        pred.append(centroids[labels[i]])

    kmeans_errors.append(mean_squared_error(true, pred))

kme = pd.DataFrame(kmeans_errors, columns=['SME'])

p = figure(title="KMeans Error by Number of Clusters",

           toolbar_location=None, tools="hover", plot_height=500)

p.circle(range(1,20),kmeans_errors,

       fill_color="#75968f", size=20, line_color="white", source=kme)

p.line(range(1,20),kmeans_errors, line_width=2, source=kme)

p.select_one(HoverTool).tooltips = [

        ('Error Rate', '@SME')

    ]



push_notebook(handle=show(p, notebook_handle=True))

####Learning Time

dat = data_left.copy().drop(['left'], axis=1)

def kmeans_act(data, size):

    train_test_split(data, test_size=0.33, random_state=42)

    kmns = KMeans(n_clusters=size).fit(data)

    df_labels = pd.DataFrame(kmns.labels_)

    print(df_labels[0].value_counts())

    df_kmns = pd.DataFrame(kmns.cluster_centers_, columns=data.columns)

    print(df_kmns)

    return kmns.labels_



dat2_kmns = dat.copy()

dat2_kmns['kmeans_label']=kmeans_act(dat.copy(), 2)



dat3_kmns = dat.copy()

dat3_kmns['kmeans_label']=kmeans_act(dat.copy(), 3)



dat4_kmns = dat.copy()

dat4_kmns['kmeans_label']=kmeans_act(dat.copy(), 4)



def kmns_plot(data, size, condition, x, y, palette):

    title= "Cluster " + str(size) + " " + x + " v " + y

    p = figure(title=title,

           toolbar_location=None, tools="hover", plot_height=500, x_axis_label= x, y_axis_label=y)

    for i in data[condition].unique().tolist():

        c0 = data[data[condition]==i]

        p.circle(c0[x], c0[y],

               fill_color=palette[i], size=10, line_color="white")



       

    push_notebook(handle=show(p, notebook_handle=True))



kmns_plot(dat2_kmns, 2, 'kmeans_label', 'average_montly_hours', 'last_evaluation', ["#75968f", "#550b1d"])    

kmns_plot(dat3_kmns, 3, 'kmeans_label', 'average_montly_hours', 'last_evaluation', ["#75968f", "#550b1d", "#c9d9d3"])    

kmns_plot(dat4_kmns, 4, 'kmeans_label', 'average_montly_hours', 'last_evaluation', ["#75968f", "#550b1d", "#c9d9d3", "#cc7878"])    

def dbscanner(data, x, y):

    dbscan = DBSCAN()

    train_test_split(data, test_size=0.33, random_state=42)

    dbscan.fit(data)

    data['dbscanned'] = dbscan.labels_

    data = data[data['dbscanned'] != -1]

    

    label_list = data['dbscanned'].unique().tolist()

    print(label_list)

    with warnings.catch_warnings():

        #warnings.filterwarnings("ignore",category=BokehUserWarning)

        p = figure(title="DBS Clusters",

               toolbar_location=None, tools="hover", plot_height=500, x_axis_label= x, y_axis_label=y)

        mapper = LinearColorMapper(Viridis256, low=label_list[0], high=label_list[len(label_list)-1])

    

        for i in label_list:

            c0 = data[data['dbscanned']==i]

            p.circle(c0[x], c0[y], source=data,

                   fill_color={'field': 'dbscanned', 'transform': mapper }, size=10, line_color="white")

        p.select_one(HoverTool).tooltips = [

            ('ClusterName', '@dbscanned')

        ]

        push_notebook(handle=show(p, notebook_handle=True)) 

    

    

dbscanner(dat.copy(), 'average_montly_hours', 'last_evaluation')