import os, sys
import pylab as plb
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from collections import Counter

# Get current work directory
cwd = os.getcwd()
cwd
# Load csv
df_1 = pd.read_csv('%s/../input/craft-cans/beers.csv' % cwd)
df_2 = pd.read_csv('%s/../input/craft-cans/breweries.csv' % cwd)

# rename unnamed column to id
df_1 = df_1.rename(columns = {"Unnamed: 0": "beer_id"})
df_2 = df_2.rename(columns = {"Unnamed: 0": "brewery_id", "name": "brewery_name"})

# Cleanout whitespaces
df_2['state'] = df_2['state'].str.strip()

# merge beers and breweries on brewery_id
brews = df_1.merge(df_2, on='brewery_id')
brews.head(5)
!{sys.executable} -m pip install wordcloud
from wordcloud import WordCloud, STOPWORDS

def explore_state(state):
    # Get dataframe of state and style column and drop NaN values
    df = brews[['state', 'style']].dropna(how='all')
    
    # If US return dataframe as is, if State input filter the styles by state
    get_data = lambda x : df['style'] if (x == 'US') else df[df.state == x]['style']
    state_style = get_data(state).dropna()
    print("Style of Beers in: %s" % state, plb.unique(state_style))
    
    # Convert dataframe to numpy
    state_style = state_style.to_numpy()
    
    # Get top 10 frequency counts of beer styles
    counts = dict(Counter(state_style).most_common(10))

    labels, values = zip(*counts.items())

    # Numpy argsort and reverse order to get descending order
    indexSort = plb.argsort(values, axis=0)[::-1]

    # get labels and values
    labels = plb.array(labels)[indexSort]
    values = plb.array(values)[indexSort]

    indexes = plb.arange(len(labels))

    # setup bar plot
    bar_width = 0.5
    
    # Get Histogram of Beer style by state or US
    plb.figure(figsize = (8, 8), facecolor = None) 
    plb.rcParams['font.family'] = 'sans-serif'
    plb.rcParams['font.sans-serif'] = 'Helvetica'
    plb.bar(indexes, values, color='gold')
    plb.xticks(indexes, labels, rotation=70)
    plb.title("Popularity of beer production in %s" % state)
    plb.xlabel('Beer Style (Descending)')
    plb.ylabel('Beer Count')
    plb.savefig('./%s_histogram.png' % state, dpi=100)
    plb.show()
    
    # Get all variations of style dropping the duplicate
    styles = plb.unique(state_style)
    
    # Set beer styles as stop words 
    style_words = []
    for style in styles:
        style = style.replace('/', '')
        style = style.split(" ")
        for s in style:
            style_words.append(s)
    
    # Remove empty values
    while("" in style_words):
        style_words.remove("")
    
    style_words = plb.asarray(style_words)
    key_words = plb.unique(style_words)
    
    # Add to Collection module's Stopword lists
    for word in key_words:
        STOPWORDS.add(word)
    
    stopwords = set(STOPWORDS)
    comment_words = ' '
    
    # If input was state, get all beer names
    # If state, get beer names from that state
    if state == "US":
        state_beers = brews['name']
    else:
        state_beers = brews[brews.state == state]['name']  

    for val in state_beers:
        # typecaste each val to string 
        val = str(val) 

        # split the value 
        tokens = val.split() 

        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 

        for words in tokens: 
            comment_words = comment_words + words + ' '

    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 

    # plot the WordCloud image                        
    plb.figure(figsize = (8, 8), facecolor = None) 
    plb.imshow(wordcloud) 
    plb.axis("off") 
    plb.tight_layout(pad = 0) 
    plb.title("Popular words in beer names in %s" % state)

    plb.show()
    
    wordcloud.to_file("./%s_beertrend.png" % state)
    print('\n')
    
explore_state('US')
explore_state('CA')
explore_state('CO')
# Get dataframe columns for state, ibu, and abv
df = brews[['state', 'ibu', 'abv']].dropna()
abvs = plb.unique(df['abv'].to_numpy())

abv_ibu = plb.empty((0,2), int)

# Create array of abv vs avg ibu
for abv in abvs:
    ibu_avg = df[df.abv == abv].ibu.mean()
    abv_ibu = plb.append(abv_ibu, [[round(abv, 3), round(ibu_avg,2)]], axis=0)

# Display abv vs ibu scatter and line
plb.style.use('dark_background')
plb.figure(figsize=(10,5), dpi=100)
plb.title('ABV vs IBU')
x = abv_ibu[:,0]*100
y = abv_ibu[:,1]

plb.plot(x, y, color="orange", label='Average IBU')
plb.xlabel('ABV (%)')
plb.ylabel('IBU')

abv = df[['abv', 'ibu']].to_numpy()

# Original data points
a = abv[:,0]*100
b = abv[:,1]

plb.scatter(a, b, s=10, alpha=0.75, color='lime', label='Data Points')
plb.legend()
plb.savefig('./abv_vs_ibu_AVG.png', dpi=100)
print("Getting Linear regression of IBU vs ABV")
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

# linregress

plb.style.use('dark_background')
plb.figure(figsize=(10,5), dpi=100)
plb.title('Linear Regression (ABV vs IBU)')
plb.plot(a, b, 'o', alpha=1, ms=5, label='original data')
plb.plot(a, intercept + slope*a, 'r', label='fitted line')
plb.xlabel('Alcohol By Volume (ABV) %')
plb.ylabel('International Bitterness Unit (IBU)')
plb.text(10.5,40, "Slope: %f\nr-squared: %f\np-value: %f\nstd_err: %f" % (slope, r_value**2, p_value, std_err), fontsize=12,  bbox={'facecolor': None, 'alpha': 0.5, 'pad': 10})

plb.legend()
plb.show
plb.savefig('./abv_vs_ibu_linregreess.png', dpi=100)
plb.close()
t_calculated = plb.linspace(2.5, 15, 40)

# slinear interpolation
linear_i = interp1d(x, y, axis=0, fill_value="extrapolate", kind='slinear')
linear = linear_i(t_calculated)

# nearest interpolation
near_i = interp1d(x, y, axis=0, fill_value="extrapolate", kind='nearest')
near = near_i(t_calculated)

# cubic interpolation
cubic_i = interp1d(x, y, axis=0, fill_value="extrapolate", kind='cubic')
cubic = cubic_i(t_calculated)

# quadratic interpolation
quadratic_i = interp1d(x, y, axis=0, fill_value="extrapolate", kind='quadratic')
quadratic = quadratic_i(t_calculated)

# plot interpolation
plb.style.use('default')
plb.figure(figsize=(10,5), dpi=100)
plb.plot(x, y, 'o', alpha=0.5, ms=5, label='data points')
plb.plot(t_calculated, linear, label='linear interpolation')
plb.plot(t_calculated, near, label='near interpolation')
# plb.plot(t_calculated, cubic, label='cubic interpolation')
# plb.plot(t_calculated, quadratic, label='quadratic interpolation')
plb.grid(True), plb.xlim(2,15), plb.ylim(0, 120)
plb.legend()
plb.title('slinear vs linear interpolation (ABV vs IBU)')
plb.xlabel('ABV (%)')
plb.ylabel('IBU')
plb.savefig('./abv_vs_ibu_interp1d.png', dpi=100)