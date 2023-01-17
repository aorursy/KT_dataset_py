import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import warnings 
from statsmodels import robust 
import warnings                                       #This sometimes does not seem to work so :
warnings.filterwarnings('ignore')

# from IPython.display import HTML
# HTML('''<script>
# code_show_err=false; 
# function code_toggle_err() {
#  if (code_show_err){
#  $('div.output_stderr').hide();                #This Script will be helpful in supressing the warnings in the notebook
#  } else {
#  $('div.output_stderr').show();
#  }
#  code_show_err = !code_show_err
# } 
# $( document ).ready(code_toggle_err);
# </script>
# To toggle on/off output_stderr, click <a href="javascript:code_toggle_err()">here</a>.''')
# # Plot color palette just to set the right kind of color scheme for our visualizarion :
# def plot_color_palette(palette: str):
#     figure = sns.palplot(sns.color_palette())
#     plt.xlabel("Color palette: " + palette)
#     plt.show(figure)

# palettes = ["deep", "muted", "pastel", "bright", "dark", "colorblind"]
# for palette in palettes:
#     sns.set(palette=palette)
#     plot_color_palette(palette)
# !pip install mplcyberpunk
# sns.set(palette='muted') #Set this according to your needs 
import mplcyberpunk
plt.style.use("cyberpunk")
mplcyberpunk.add_glow_effects()

HSD = pd.read_csv('../input/habermans-survival-data-set/haberman.csv', header=None, names=['age', 'year', 'nodes', 'status'])
HSD.info()
HSD.columns
HSD["status"].value_counts()
HSD.describe()
HSD.plot(kind = "scatter", x = 'nodes', y = 'age')
plt.title("Number of positive axillary nodes detected vs. Age of Patient ")
plt.show()
warnings.filterwarnings('ignore')

# sns.set_style("whitegrid")
sns.FacetGrid( HSD, hue = "status") \
    .map(plt.scatter, "age", "nodes")\
    .add_legend()
plt.title("Number of positive axillary nodes detected vs. Age of Patient ")
plt.show()



plt.close()
# sns.set_style("whitegrid")
# plt.suptitle("Pair plot")
g = sns.pairplot(HSD, vars = ["age", "year", "nodes"],  hue = "status", size = 4)
g.fig.suptitle("Pair Plot", fontsize=30)
# g.fig.show()
plt.show()
survived = HSD[HSD["status"] == 1]
dead = HSD[HSD["status"] == 2]

print(survived["age"].describe())
print(dead["age"].describe())
print("Medians :")
print("Median age of the people that survived : ",np.median(survived["age"]))
print("Median age of the people that could not survive : ", np.median(dead["age"]))
print("Median Positive lymph nodes in the people that survived : ", np.median(survived["nodes"]))
print("Median Positive lymph nodes in the people that could not survive :  ", np.median(dead["nodes"]))
print("------------------------------------------------------------------------------------------------")
print("Quantiles :")
print("Survived : ")
print("AGE :",np.percentile(survived["age"], np.arange(0, 100, 25)))
print("NODES : ", np.percentile(survived["nodes"], np.arange(0,100,25)))
print("Dead : ")
print("AGE :",np.percentile(dead["age"], np.arange(0, 100, 25)))
print("NODES : ", np.percentile(dead["nodes"], np.arange(0,100,25)))
print("------------------------------------------------------------------------------------------------")
print("Percentiles : ")
print("Survived : ")
print("AGE :",np.percentile(survived["age"], 40))
print("NODES : ", np.percentile(survived["nodes"], 40))
print("dead : ")
print("AGE :",np.percentile(dead["age"], 40))
print("NODES : ", np.percentile(dead["nodes"], 40))
print("------------------------------------------------------------------------------------------------")
print("Median Abolute Deviation : ")
print("Survived :")
print("AGE :",robust.mad(survived["age"]))
print("NODES :",robust.mad(survived["nodes"]))
print("Dead :")
print("AGE :",robust.mad(dead["age"]))
print("NODES :",robust.mad(dead["nodes"]))

plt.plot(survived["age"], np.zeros_like(survived["age"]), 'o')
plt.title("survival status univarite scatterplot")
plt.plot(dead["age"], np.zeros_like(dead["age"]), 'r')
plt.show()
warnings.filterwarnings("ignore")
sns.FacetGrid(HSD, hue = "status", size = 6)\
    .map(sns.distplot, "age")\
    .add_legend()
plt.title('Histogram of ages of patients', fontsize=30)
plt.show()
sns.FacetGrid(HSD, hue = "status", size = 6)\
    .map(sns.distplot, "nodes")\
    .add_legend()
plt.title('Histogram of status of patients', fontsize=30)
plt.show()
counts,bin_edges = np.histogram(survived["age"], bins = 10, density = True )
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges, "\n", "#"*90)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.gca().legend(('PDF','CDF'))
plt.title('PDF and CDf of ages of patients who survived', fontsize=30)
plt.show()


counts,bin_edges = np.histogram(dead["age"], bins = 10, density = True )
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges, "\n", "#"*90)
cdf = np.cumsum(pdf)
plt.title('PDF and CDf of ages of patients who could not survive', fontsize=30)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.gca().legend(('PDF','CDF'))
plt.show()


counts,bin_edges = np.histogram(survived["age"], bins = 10, density = True )
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

counts,bin_edges = np.histogram(dead["age"], bins = 10, density = True )
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

plt.gca().legend(('PDF of survived','CDF of survived ', 'PDF of dead ', 'CDF of dead'))
plt.title('PDF and CDf of ages of both, survived and dead patients', fontsize=30)

plt.show()





sns.violinplot(x = 'status', y = 'age', data = HSD)
plt.title("Box Plot of status of patients who could and couldn't survive", fontsize=30)
plt.show()
sns.boxplot(x = "status" , y = "age", data = HSD )
plt.title("Box Plot of status of patients who could and couldn't survive", fontsize=30)
plt.show()
sns.boxplot(x = 'status', y = 'nodes', data = HSD)
plt.title("Box Plot of nodes of patients who could and couldn't survive", fontsize=30)
plt.show()