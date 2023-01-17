from pandas import read_csv

data = read_csv('../input/glass.csv')

target="Type"

X = data.drop(target,1)

y = data[target]
features = X.columns
from IPython.display import display

from scipy.stats import kruskal, f_oneway, ttest_ind

from matplotlib import colors, legend, patches

from numpy import float32

from pandas import DataFrame

from seaborn import pointplot, boxplot, cubehelix_palette, set_style, kdeplot, color_palette, heatmap, diverging_palette

from matplotlib.pyplot import show, figure, rc_context, subplot, suptitle, title

from itertools import permutations

from scipy.stats import ttest_ind, mannwhitneyu, ranksums

from statsmodels.stats.weightstats import ztest



# ===  Separate data rows by a category, then return as a dictionary === #



def get_samples(data, category, numeric):

    samples = {}

    for c in data[category].unique():

        key = category + "_" + str(c)

        series = data[data[category] == c][numeric]

        samples[key] = series.rename(str(c))

    return samples



# === Return a dataframe of F-Oneway & Kruskal scores === #



def different_means_test_for_groups(dataframe,category,numeric):

    

    samples = get_samples(dataframe,category,numeric)

    

    if type(samples) == dict:

        output = []

        for key in samples.keys(): output += [samples[key]]

        samples = output

    

    scores = {"kruskal" : kruskal(*samples), 

              "f_oneway" : f_oneway(*samples)}

    

    return DataFrame(scores, index=("statistic","p-value")).T.round(2)



# === Produce a p-value grid for parametric different means === #



def parametric_different_means_p_value_grid(samples):

    

    p_value_table = DataFrame(index = samples.keys(), columns = samples.keys())

        

    for c1,c2 in permutations(samples.keys(),2):

                

        if (len(samples[c1]) > 30 and len(samples[c2]) > 30):

            z, p = ztest(samples[c1], samples[c2])

        else:

            t, p = ttest_ind(samples[c1],samples[c2])

        

        p_value_table[c1][c2] = round(p,2)

    

    df = p_value_table.fillna(float32(None))

    heat_map = heatmap(df, annot=True, annot_kws={"size" : 24}, linewidths=2, cmap=colors.ListedColormap([diverging_palette(10, 220, sep=80, n=10)[4]]), cbar=False, square=True)

    heatmap(df, mask = df > 0.05, cmap=colors.ListedColormap([diverging_palette(10, 220, sep=80, n=10)[8]]), annot=True, annot_kws={"size" : 24}, linewidths=2, cbar=False, square=True)

    title("diffmeans p-value Grid", fontsize=14, loc="left")



    p = diverging_palette(10, 220, sep=80, n=10)

    myColors = [p[8],p[4]]

    classes = ['p < 0.05', 'p >= 0.05']

    recs = []

    for i in range(0,len(myColors)):

        recs.append(patches.Rectangle((0,0),1,1,fc=myColors[i]))

    heat_map.legend(recs, classes, loc=4, bbox_to_anchor=(1, 1))

    

# === Produce a p-value grid for ranksums === #



def ranksums_p_value_grid(samples):

    

    p_value_table = DataFrame(index = samples.keys(), columns = samples.keys())

        

    for c1,c2 in permutations(samples.keys(),2):

                

            statistic, p = ranksums(samples[c1],samples[c2])

        

            p_value_table[c1][c2] = round(p,2)

    

    df = p_value_table.fillna(float32(None))

    heat_map = heatmap(df, annot=True, annot_kws={"size" : 24}, linewidths=2, cmap=colors.ListedColormap([diverging_palette(10, 220, sep=80, n=10)[4]]), cbar=False, square=True)

    heatmap(df, mask = df > 0.05, cmap=colors.ListedColormap([diverging_palette(10, 220, sep=80, n=10)[8]]), annot=True, annot_kws={"size" : 24}, linewidths=2, cbar=False, square=True)

    title("ranksums p-value Grid", fontsize=14, loc="left")



    p = diverging_palette(10, 220, sep=80, n=10)

    myColors = [p[8],p[4]]

    classes = ['p < 0.05', 'p >= 0.05']

    recs = []

    for i in range(0,len(myColors)):

        recs.append(patches.Rectangle((0,0),1,1,fc=myColors[i]))

    heat_map.legend(recs, classes, loc=4, bbox_to_anchor=(1, 1))



# === Produce a p-value grid for mannwhitneyu === #



def mannwhitneyu_p_value_grid(samples):

    

    p_value_table = DataFrame(index = samples.keys(), columns = samples.keys())

        

    for c1,c2 in permutations(samples.keys(),2):

                

        statistic, p = mannwhitneyu(samples[c1],samples[c2])

        

        p_value_table[c1][c2] = round(p,2)

    

    df = p_value_table.fillna(float32(None))

    heat_map = heatmap(df, annot=True, annot_kws={"size" : 24}, linewidths=2, cmap=colors.ListedColormap([diverging_palette(10, 220, sep=80, n=10)[4]]), cbar=False, square=True)

    heatmap(df, mask = df > 0.05, cmap=colors.ListedColormap([diverging_palette(10, 220, sep=80, n=10)[8]]), annot=True, annot_kws={"size" : 24}, linewidths=2, cbar=False, square=True)

    title("mannwhitneyu p-value Grid", fontsize=14, loc="left")



    p = diverging_palette(10, 220, sep=80, n=10)

    myColors = [p[8],p[4]]

    classes = ['p < 0.05', 'p >= 0.05']

    recs = []

    for i in range(0,len(myColors)):

        recs.append(patches.Rectangle((0,0),1,1,fc=myColors[i]))

    heat_map.legend(recs, classes, loc=4, bbox_to_anchor=(1, 1))



# === Display full set of category x numerical analyses === #



def display_multi_category_x_numeric_analysis(data, category, numeric):

    

    # === Use only non-null values in numeric/category === #

    

    data = data[(data[category].notnull()) | (data[numeric].notnull())]

    

    # === Set up visualization attributes



    set_style("whitegrid")

    

    subcategory_count = len(data[category].unique())

    chosen_palette = color_palette("colorblind",subcategory_count, desat=0.85)



    # === Set up axes attributes === #

    

    print(numeric)

    

    figure(figsize=(12.5,12.5))

    #suptitle(numeric, y=0.94, fontsize=16)



    # === Order category values by mean === #

    

    order = data.groupby(category)[numeric].mean().sort_values(ascending=False).index

    

    # === Get samples === #



    samples = get_samples(data, category, numeric)



    # === The Visualizations === #



    with rc_context({'lines.linewidth': 0.8}):

    

        # === Box Plot === #



        cell_0_1 = subplot(222)

        

        box_plot = boxplot(x=category, y=numeric, order=order, palette = chosen_palette, data=data)

        

        # === Labels

        

        box_plot.set_xlabel(box_plot.get_xlabel, visible=False)

        box_plot.set_ylabel(numeric + " (mg)", visible=False)

        box_plot.set_xticklabels(box_plot.get_xticklabels())

        

        # === Point Plot === #

        

        subplot(221, sharey=cell_0_1)

        point_plot = pointplot(x=category, y=numeric, order=order, data=data, capsize=.14, color="#383838")

        

        # == Labels

        

        point_plot.set_ylabel(numeric + " mean")

        point_plot.set_xlabel(point_plot.get_xlabel(), visible=False)

        point_plot.set_xticklabels(point_plot.get_xticklabels())

        

        # === Ranksums Grid === #

        

        subplot(223)

        

        ranksums_p_value_grid(samples)

        

        # === Different Means Grid === #

    

        subplot(224)

        

        parametric_different_means_p_value_grid(samples)

    

        show()

    

        if len(data[category].unique()) > 2:

            display(different_means_test_for_groups(data,category,numeric))
from itertools import combinations, permutations

from math import log1p,tanh, exp



for x,y in combinations(features.tolist(),2):

    data[" * ".join([x,y])] = data[x] * data[y]

    

total = [0] * len(data)

for feature in features: total += data[feature]

data["Total"] = total

    

for feature in features:

    

    data[feature + " Percentage"] = data[feature] / total

    

    data[feature + "^2"] = data[feature] ** 2

    data[feature + "^3"] = data[feature] ** 3

    data[feature + "^0.5"] = data[feature] ** 0.5

    

    data["log(%s)" % feature] = data[feature].apply(log1p)

    data["exp(%s)" % feature] = data[feature].apply(exp)

    #data["tanh(%s)" % feature] = data[feature].apply(tanh)

    

    try:

        data["1/%s" % feature] = data[feature].apply(lambda x: 1 / x)

    except:

        pass



for x,y in permutations(features,2):

    if (data[x] > 0).all() and (data[y] > 0).all():

        data[" / ".join([x,y])] = data[x] / data[y]

        
data = data.dropna(axis=1)
new_features = [feature for feature in data.columns if feature not in features]

new_features.remove("Type")
# === List counts of statistically significant differences === #



table = {}



for feature in features.tolist() + new_features:

    

    table[feature] = {}



    for value in data[target].unique():



        # === Get other type values === #



        temp = data[target].unique().tolist()

        temp.remove(value)



        # === Set up data group by value === #



        a = data[data[target] == value]

        

        table[feature][value] = 0



        for b in temp:

            

            b = data[data[target] == b]

            

            statistic, p = ranksums(a[feature],b[feature])

            

            if p < 0.05:

                

                table[feature][value] +=1

                

        table[feature][value] /= len(data[target].unique()) - 1
display(DataFrame(table).T.sort_values([1,2,3],ascending=False).head(70))
table = {}



for feature in features.tolist() + new_features:

    

    table[feature] = {}



    for a,b in combinations(data[target].unique(), 2):



        aa = data[data[target] == a]

        bb = data[data[target] == b]

            

        statistic, p = ranksums(aa[feature],bb[feature])



        if p < 0.05:



            table[feature]["[%s][%s]" % (a,b)] = 1



        else:

            

            table[feature]["[%s][%s]" % (a,b)] = 0
DataFrame(table).mean().sort_values(ascending=False).head(70)
DataFrame(table).T.mean().sort_values(ascending=False)
DataFrame(table).T["[1][3]"].sort_values(ascending=False).head()
for new in new_features: display_multi_category_x_numeric_analysis(data,target,new)