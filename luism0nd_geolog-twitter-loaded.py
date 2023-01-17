import pandas as pd
import geopandas
from shapely.geometry import Point
from gensim.models import LsiModel
from gensim import similarities, corpora
with open('../input/es-stopwords/spanish_stopwords.txt', 'r', encoding='utf8') as f:
    stops = f.read()
stops_es = stops.split('\n')

csv_data = pd.read_csv('../input/world-cities-in-twitter-dataset2/world_cities_in_twitter_dataset2.csv')

csv_data['geometry'] = list(zip(csv_data.lng, csv_data.lat))
csv_data['geometry'] = csv_data['geometry'].apply(Point)
csv_data = geopandas.GeoDataFrame(csv_data, geometry='geometry')
csv_data.head()
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
base = world.plot(color='black', edgecolors='white', figsize=(65,95))
base.set_facecolor('black')
print('All cities in twitter dataset. No data for Venezuela, Uruguay.')
df = csv_data.plot(ax=base, column="city")
data_scope = (49.699636, -123.73046, 18.25, -55.9541)
maxy, minx, maxx, miny = data_scope
df.set_xlim(minx, maxx)
df.set_ylim(miny, maxy)
csv_data.head()

#df.loc[df['column_name'] == some_value]
mx = csv_data.loc[csv_data['iso2'] == 'mx']

def strip_punct(line):
    charset = set()
    for ch in line:
        charset.update(ch)
    punct = [ch for ch in charset if not ch.isalpha()]# and not ch.isdigit()]
    if ' ' in punct:
        punct.remove(' ')
    for ch in punct:
        line = line.replace(ch, ' ').lower()
        line = line.replace('  ', ' ').lower()
    return line

def remove_stops(line):
    line = line.split()
    return [w for w in line if not w in stops_es and not w.isdigit() and len(w)>2] 
name = 'Top500Twitter_es'
dictionary = corpora.Dictionary.load('../input/geolog/'+name+'.dict')
corpus = corpora.MmCorpus('../input/geolog/'+name+'.mm')
lsi = LsiModel.load('../input/geolog/'+name+'_model.lsi')
headers = pd.read_csv('../input/labels/top500twitter_headers.csv')
headers = headers.iloc[:,1].tolist() 

#TEST!# #For test lines, lets take our top 10,000 words hispa sub reddits data set
hispa_reddits = pd.read_csv('../input/test-hispa-reddits/hispa_reddits_top_10000_tokens.csv')

test_lines = hispa_reddits['top tokens'].tolist()
subreddits = hispa_reddits['country'].tolist()
test_lines = [t.replace(',',' ') for t in test_lines]
hispa_reddits_zip = list(zip(subreddits,test_lines))
hispa_reddits_lines = [x[0]+' '+x[1] for x in hispa_reddits_zip]

def get_map(test_line):
    test_line = test_line.split()
    subname = test_line[0]
    test_line = test_line[:5000]
    test_line_ = test_line[1:100]

    # REMOVE STOP WORDS AND PUNCTUATION
    #test_line = strip_punct(test_line)
    #test_line = remove_stops(test_line)

    # TRANSFORM TEST LINE
    query_bow = dictionary.doc2bow(test_line)
    query_lsi = lsi[query_bow] # convert the query to lsi space

    # GET SIMILARITY SCORES
    index = similarities.MatrixSimilarity.load('../input/geolog/'+name+'_similarity.index')
    similarity_scores = index[query_lsi]
    similarity_scores_headers = list(zip(similarity_scores,headers))
    similarity_scores_headers = [list(y) for y in similarity_scores_headers]
    sim_score_headers_df = pd.DataFrame(similarity_scores_headers,columns=['score', 'city'])
    csv_data1 = csv_data.merge(sim_score_headers_df, on='city')
    n = 15
    csv_data2 = csv_data1.nlargest(n, 'score')
    csv_data2['w'] = list(sorted([x/n+1 for x in range(n)],reverse=True))
    csv_data2['z'] = csv_data2['w'] * csv_data2['w'] * csv_data2['w'] * csv_data2['w'] * 200

    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    #test_line_ = ' '.join(test_line_)
    test_line_str = list(split(test_line_,11))
    test_line_str = [' '.join(t) for t in test_line_str]
    test_line_str = '\n'.join(test_line_str)
    #test_line_str
    csv_data2.sample()

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    base = world.plot(color='#0f0f0f', edgecolors='white', figsize=(35,35))
    base.set_facecolor('#161616')
    df = csv_data2.plot(ax=base, column="score", markersize='z', cmap='Reds')#,figsize=(35,35))

    data_scope = (49.699636, -123.73046, 9.25, -55.9541)
    maxy, minx, maxx, miny = data_scope
    props_t = dict(boxstyle='round', facecolor='#e0e0e0', alpha=.9)
    props = dict(boxstyle='round', facecolor='white', alpha=.9)


    country_subs = hispa_reddits['country'].tolist()
    country_sub = '100 palabras en español más frecuentes del subreddit: r/'+subname

    textstr = 'Palabras más frecuentes en español de subreddits de habla hispana \n y las ciudades con vocabularios más similares'#.upper()
    country_sub_cities = 'Ciudades con vocabularios similares:*'#.upper()
    #disclaimer = ''
    source = '*Obtenido de una comparación con datos de Twitter \n Fuente: The top-5000 frequent Spanish words in Twitter for 331 cities in the Spanish-speaking world \n https://www.datos.gov.co/Ciencia-Tecnolog-a-e-Innovaci-n/The-top-5000-frequent-Spanish-words-in-Twitter-for/nmid-inr9'
    source1 = 'Fuente: Top 10000 frequent Spanish words in hispanic subreddits for 21 countries (October 2018) \n https://www.kaggle.com/luism0nd/test-hispa-reddits'
    author = 'Autor: u/serioredditor'

    cities_r = csv_data2['city'].tolist()
    cities_r = [c.replace('_',' ').title() for c in cities_r]
    cities_r = '\n'.join(cities_r)
    #csv_data2.drop(['Unnamed: 0'],axis=1)
    #df.legend(csv_data2, ['Similitud'])

    df.text(0.43, .99, textstr, fontsize=28,
            verticalalignment='top',bbox=props_t, transform=df.transAxes, )
    df.text(0.45, 0.92, author, fontsize=20,
            verticalalignment='top',bbox=props, transform=df.transAxes)
    df.text(0.45, 0.89, source1, fontsize=16,
            verticalalignment='top',bbox=props, transform=df.transAxes, )

    df.text(0.43, 0.84, country_sub, fontsize=28,
            verticalalignment='top',bbox=props_t, transform=df.transAxes, )
    df.text(0.45, 0.80, test_line_str, transform=df.transAxes, fontsize=19,
            verticalalignment='top',bbox=props)

    df.text(0.02, 0.64, country_sub_cities, transform=df.transAxes, fontsize=26,
            verticalalignment='top',bbox=props_t)
    df.text(0.04, 0.60, cities_r, transform=df.transAxes, fontsize=24,
            verticalalignment='top',bbox=props)
    df.text(0.04, 0.05, source, fontsize=16,
            verticalalignment='top',bbox=props, transform=df.transAxes)


    minx, miny, maxx, maxy = csv_data2.total_bounds
    p = 4
    df.set_xlim(minx-p, maxx+p)
    df.set_ylim(miny-p, maxy+p)
    print(textstr)
    print(' '.join(test_line_))
    #df.legend()
    fig = df.get_figure()
    fig.savefig(subname+"_map.png")


#hispa_reddits_lines = hispa_reddits_lines[:2]
for l in hispa_reddits_lines:
    get_map(l)
