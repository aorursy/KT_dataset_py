import os, warnings, folium

import numpy as np

import pandas as pd

import plotly.graph_objs as go

import matplotlib.pyplot as plt

from geopy.distance import great_circle

from shapely.geometry import MultiPoint

from sklearn.cluster import DBSCAN

from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import pairwise_distances, mean_squared_error

from sklearn.preprocessing import minmax_scale, MultiLabelBinarizer

from sklearn.decomposition import NMF

from random import randint

from download_csv_file import create_download_link

from ml_metrics import mapk, apk



# Pandas and Numpy configs

pd.set_option('display.max_columns', 20)

np.set_printoptions(suppress=True)



# Input data files are available in the "../input/" directory.

main_path = '../input'

    

# Init ploty in offline mode

plot_template = 'plotly_white'



warnings.filterwarnings('ignore')



print(os.listdir(main_path))
### Path of file to read

data_file_path = f'{main_path}/flickr_london/london_20k.csv'



# Change data types

data_type = {

    'photo_id': 'object',

    'owner': 'object',

    'faves': 'float16',

    'lat': 'float32',

    'lon': 'float32',

    'taken': 'datetime64'

}



# Read file into a variable data

raw = pd.read_csv(data_file_path, 

                  engine='python', 

                  sep=',', 

                  encoding='utf-8', 

                  dtype=data_type, 

                  decimal=',')

data_dim = raw.shape



print(f'Dataframe dimentions: {data_dim}', f'\n{"-"*50}\nData Types:\n{raw.dtypes}')



# Show head

raw.head()
# Find total missing values

data = raw[['photo_id','owner','lat','lon','taken']]

missing_nan = data.isna().sum()



print('TOTAL MISSINGS:', missing_nan, sep='\n')



# Remove missing values

data = data.dropna(subset=['lat','lon'])

new_size = len(data.index)

print(f'{"-"*50}\n{data_dim[0]-new_size} empty rows are removed.')
# Create dataframe filled with DBSCAN params and clusters

def paramsClusters(data, eps_range, minPts_range):

    m_per_rad = 6371.0088 * 1000

    df = pd.DataFrame(columns=['eps','min_pts','num_clusters'])

    for m in minPts_range:

        for e in eps_range:

            eps_rad = e/m_per_rad

            eps_rad = eps_rad

            #  DBSCAN based on Haversine metric

            db = DBSCAN(eps=eps_rad, min_samples=m, algorithm='ball_tree', metric='haversine').fit(np.radians(data[['lat','lon']]))

            c = len(set(db.labels_ + 1))

            df = df.append({'eps': e, 'min_pts': m, 'num_clusters': c}, ignore_index=True)

    

    return df



# DBSCAN trend - epsilons and clusters

epsilons = list(map(lambda n: n*20, range(1,16)))

min_points = list(map(lambda n: n*10, range(1,6)))

pc = paramsClusters(data, epsilons, min_points)

EVC = go.Figure()



for m in pc.min_pts.unique():

    df = pc[pc.min_pts == m]

    EVC.add_trace(go.Scatter(

        x=df.eps,

        y=df.num_clusters,

        name=f'Min Samples: {m} points',

        mode='lines+markers',

        marker=dict(size=8),

        line=dict(width=2),

        line_shape='spline'

    ))

    

EVC.update_layout(

    title='The number of detected clusters with different valuses of MinSamples',

    xaxis=dict(title='Epsilon', zeroline=False, dtick=40),

    yaxis=dict(title='Number of Cluster', zeroline=False),

    template=plot_template

)



# DBSCAN trend - samples and clusters

epsilons = list(map(lambda n: n*20, range(1,5)))

min_points = list(map(lambda n: n*5, range(1,11)))

pc = paramsClusters(data, epsilons, min_points)

MVC = go.Figure()

    

for e in pc.eps.unique():

    df = pc[pc.eps == e]

    MVC.add_trace(go.Scatter(

        x=df.min_pts,

        y=df.num_clusters,

        name=f'Epsilon: {e} m',

        mode='lines+markers',

        marker=dict(size=8),

        line=dict(width=2),

        line_shape='spline'

    ))



MVC.update_layout(

    title='The number of detected clusters with different valuses of Eps',

    xaxis=dict(title='The number of samples in neighborhood', zeroline=False, dtick=5),

    yaxis=dict(title='Number of Cluster', zeroline=False),

    template=plot_template

)



EVC.show()

MVC.show()
# Calculate DBSCAN based on Haversine metric    

def HDBSCAN(df, epsilon, minPts, x='lat', y='lon'):

    

    # Find most centered sample in a cluster

    def getCenterMostPts(cluster):

        centroid = (MultiPoint(cluster.values).centroid.x, MultiPoint(cluster.values).centroid.y)

        centermost_point = min(cluster.values, key=lambda point: great_circle(point, centroid).m)

        return tuple(centermost_point)



    m_per_rad = 6371.0088 * 1000

    eps_rad = epsilon/m_per_rad

    photo_coords = df.loc[:, {x,y}]

    photo_coords = photo_coords[['lat','lon']]

    db = DBSCAN(eps=eps_rad, min_samples=minPts, algorithm='ball_tree', metric='haversine').fit(np.radians(photo_coords))

    cluster_labels = db.labels_ + 1

    num_clusters = len(set(cluster_labels))



    # Put clusters and their subset of coords in an array

    clusters = pd.Series([photo_coords[cluster_labels==n] for n in range(num_clusters)])



    # Find centroid of each cluster

    centroids = clusters.map(getCenterMostPts)

    

    # Pull rows from original data frame where row numbers match the clustered data

    rows = clusters.apply(lambda c: c.index.values)

    clustered_df = rows.apply(lambda row_num: df.loc[row_num])

    

    # Append cluster numbers and centroid coords to each clustered dataframe

    lats,lons = zip(*centroids)

    new_df = []

    for i, v in clustered_df.iteritems():

        v.loc[:, 'cluster_num'] = i

        v.loc[:, 'cent_lat'] = lats[i]

        v.loc[:, 'cent_lon'] = lons[i]

        new_df.append(v)    

    new_df = pd.concat(new_df)

    

    return new_df

    

cdata = HDBSCAN(data, epsilon=120, minPts=10)

print(f'Number of clusters: {len(cdata.cluster_num.unique())}')
# Convet matplotlib colormap to plotly

def matplotlibToPlotly(cmap, pl_entries):

    h = 1.0/(pl_entries-1)

    pl_colorscale = []

    

    for k in range(pl_entries):

        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))

        pl_colorscale.append('rgb'+str((C[0], C[1], C[2])))

        

    return pl_colorscale





# Show plot

unique_labels = cdata.cluster_num.unique()

colors = matplotlibToPlotly(plt.cm.Spectral, len(unique_labels))

DB = go.Figure()

leaflet_map = folium.Map(location=[51.514205,-0.104371], zoom_start=12, tiles='Cartodb Positron')



for k,col in zip(unique_labels, colors):

    # Check if label number is 0, then create noisy points 

    if k == 0:

        col = 'gray'

        df = cdata[cdata.cluster_num == 0]

        

        DB.add_trace(go.Scatter(

            x=df.lat,

            y=df.lon,

            mode='markers',

            name='noise',

            marker=dict(size=3, color=col),

            hoverinfo='none'

        ))

        

    # Check the remaining clusters

    else:

        col = col

        df = cdata[cdata.cluster_num == k]

        lat = df.lat

        lon = df.lon

        cent_lat = df.cent_lat.unique()

        cent_lon = df.cent_lon.unique()

        

        # Bokeh plot

        DB.add_trace(go.Scatter(

            x=lat,

            y=lon,

            mode='markers',

            name='point',

            marker=dict(size=5, color=col),

            text=df.photo_id.apply(lambda id: f'photo_id: {id}'),

            hoverinfo='none',

            showlegend=False

        ))

        DB.add_trace(go.Scatter(

            x=cent_lat,

            y=cent_lon,

            mode='markers',

            name='centroid',

            text=f'cluster: {k}',

            marker=dict(

                size=12,

                color=col,

                line=dict(color='gray', width=1)

            ),

            hoverinfo='x+y+name+text'

        ))

        

        # Map plot

        folium.Marker(

            location=[cent_lat, cent_lon],

            icon=folium.Icon(icon='map-marker')

        ).add_to(leaflet_map)

        

        

DB.update_layout(

    title='DBSCAN based on Haversine including center most points',

    hovermode='closest',

    showlegend=False,

    xaxis=dict(title='Latitude', zeroline=False),

    yaxis=dict(title='Longitude', zeroline=False),

    template=plot_template

)



DB.show()

leaflet_map
# Remove noise cluster from the training set

clean_data = cdata[cdata.cluster_num!=0]



# Distribution plot

def chunk(seq, size):

    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



distrib_df = clean_data.groupby(['cluster_num'])['cluster_num'].count().reset_index(name='photo_num')

chunked_distrib = chunk(distrib_df, 50)



for df in chunked_distrib:

    cluster_distrib = go.Figure([go.Bar(x=df.cluster_num, y=df.photo_num)])

    cluster_distrib.update_layout(

        title='Distribution of photos among clusters',

        xaxis=dict(title='Cluster id', dtick=1),

        yaxis=dict(title='Number of images'),

        template=plot_template

    )

    cluster_distrib.show()
# Find most frequent string in array

def mostFreqStr(array):

    array = [i for i in array if str(i) != 'nan']

    if len(array) != 0:

        counts = np.unique(array, return_counts=True)[1]

        max_index = np.argmax(counts)

        freq_bin = array[max_index]

        return freq_bin

    else:

        return np.nan



# Find median of array included Timestamps

def medTimestamps(array):

    if len(array) == 1:

        return array[0]

    else:

        if len(array) % 2 == 0:

            delta = array[int(len(array)/2)] - array[int(len(array)/2-1)]

            median = pd.Timestamp(array[int(len(array)/2-1)] + delta)

        else:

            time = pd.Timestamp(array[int(len(array)/2)]).time()

            ser = pd.Series(array)

            date = pd.Timestamp.fromordinal(int(ser.apply(lambda x: pd.to_datetime(x).toordinal()).median(skipna=True))).date()

            median = pd.Timestamp.combine(date,time)     

        return median



# Create database of locations

POI = pd.DataFrame(columns=['location_id', 'user_id', 'lat', 'lon', 'visit_time'])

threshold = np.timedelta64(6, 'h')



for i,g in clean_data.groupby(by='cluster_num'):

    l = {}

    l['location_id'] = randint(100000,999999)

    l['lat'] = g.cent_lat.unique()[0]

    l['lon'] = g.cent_lon.unique()[0]

    

    for u in g.owner.unique():

        l['user_id'] = u

        taken = g.loc[g.owner == u, 'taken'].sort_values()

        t_indices = taken.keys()

        t_values = taken.values

        visit_times = []

        

        if len(t_values) == 1:

            l['visit_time'] = pd.Timestamp(t_values[0])

            POI = POI.append(l, ignore_index=True)

        

        else:

            for t in range(1, len(t_values)):

                if t_values[t]-t_values[t-1] < threshold:

                    visit_times.append(t_values[t-1])

                else:

                    visit_times.append(t_values[t-1])

                    l['visit_time'] = medTimestamps(visit_times)

                    POI = POI.append(l, ignore_index=True)

                    visit_times = []



display(POI.head(10))

                    

# Create a lint to download

create_download_link(POI, filename='prefiltered.csv')
# Path of file to read

prefiltered_file_path = f'{main_path}/flickr_london_prefiltered/prefiltered.csv'



# Change data types

data_type = {

    'faves': 'float16',

    'lat': 'float32',

    'lon': 'float32',

    'visit_time': 'datetime64'

}



# Read csv file and convert it to a Multiindex

LPD = pd.read_csv(prefiltered_file_path, engine='python', sep=',', encoding='utf-8', dtype=data_type, decimal=',')

LPD = LPD.set_index(keys=['user_id', 'location_id'])

display(LPD.head(10))



# Split dataset

visit_limit = LPD.groupby(level=[0,1])['visit_time'].count()

visit_limit = visit_limit[visit_limit>3]

mask = LPD.index.isin(visit_limit.index) == True

X = LPD[mask]

y = X.index.get_level_values(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=70)
# Find ratings

train_rating = X_train.groupby(['location_id','user_id'])['visit_time'].count().reset_index(name='rating')

train_rating.head(10)
def normalize(df):

    # Normalize number of visit into a range of 1 to 5

    df['rating'] = minmax_scale(df.rating, feature_range=[1,5])

    return df



r_df = normalize(train_rating)



# Create a rating matrix

r_df = train_rating.pivot_table(

    index='user_id', 

    columns='location_id', 

    values='rating', 

    fill_value=0

)

    

# Calculate the sparcity percentage of matrix

def calSparcity(m):

    m = m.fillna(0)

    non_zeros = np.count_nonzero(m)/np.prod(m.shape) * 100

    sparcity = 100 - non_zeros

    print(f'The sparcity percentage of matrix is %{round(sparcity,2)}')



display(r_df.head())

calSparcity(r_df)
# Create user-user similarity matrix

def improved_asym_cosine(m, mf=False,**kwarg):

    # Cosine similarity matrix distance

    cosine = cosine_similarity(m)



    # Asymmetric coefficient

    def asymCo(X,Y):

        co_rated_item = np.intersect1d(np.nonzero(X),np.nonzero(Y)).size

        coeff = co_rated_item / np.count_nonzero(X)

        return coeff

    asym_ind = pairwise_distances(m, metric=asymCo)



    # Sorensen similarity matrix distance

    sorensen = 1 - pairwise_distances(np.array(m, dtype=bool), metric='dice')



    # User influence coefficient

    def usrInfCo(m):

        binary = m.transform(lambda x: x >= x[x!=0].mean(), axis=1)*1

        res = pairwise_distances(binary, metric=lambda x,y: (x*y).sum()/y.sum() if y.sum()!=0 else 0)

        return res       

    usr_inf_ind = usrInfCo(m)



    similarity_matrix = np.multiply(np.multiply(cosine,asym_ind),np.multiply(sorensen,usr_inf_ind))



    usim = pd.DataFrame(similarity_matrix, m.index, m.index)

    

    # Check if matrix factorization was True

    if mf:

        # Binary similarity matrix

        binary = np.invert(usim.values.astype(bool))*1

        model = NMF(**kwarg)

        W = model.fit_transform(usim)

        H = model.components_

        factorized_usim = np.dot(W,H)*binary + usim

        usim = pd.DataFrame(factorized_usim, m.index, m.index)

                

    return usim



s_df = improved_asym_cosine(r_df)

display(s_df.head())

calSparcity(s_df)
# Find probability of contexts

contexts = X_train.filter(['season','daytime','weather']).apply(lambda x: (x.season,x.daytime,x.weather), axis=1).reset_index(name='context')

IF = contexts.groupby(['location_id','context'])['context'].count()/contexts.groupby(['context'])['context'].count()

IDF = np.log10(contexts.groupby(['location_id','user_id'])['user_id'].count().sum()/contexts.groupby(['location_id'])['user_id'].count())

contexts_weight = (IF * IDF).to_frame().rename(columns={0: 'weight'})



# Create a context-location matrix

lc_df = contexts_weight.pivot_table(

    index='context', 

    columns='location_id', 

    values='weight',

    fill_value=0

)





display(lc_df.head())

calSparcity(lc_df)
cs_df = pd.DataFrame(cosine_similarity(lc_df), index=lc_df.index, columns=lc_df.index)

display(cs_df.head())

calSparcity(cs_df)
def CF(user_id, location_id, s_matrix):

    r = np.array(r_df)

    s = np.array(s_matrix)

    users = r_df.index

    locations = r_df.columns

    l = np.where(locations==location_id)[0]

    u_idx = np.where(users==user_id)[0]

        

    # Means of all users

    means = np.array([np.mean(row[row!=0]) for row in r])

    

    # Check if l is in r_rating

    if location_id in r_df:

        # Find similar users rated the location that target user hasn't visited

        idx = np.nonzero(r[:,l])[0]

        sim_scores = s[u_idx,idx].flatten()

        sim_users = zip(idx,sim_scores)

    

        # Check if there is any similar user to target user

        if idx.any():

            sim_ratings = r[idx,l]

            sim_means = means[idx]

            numerator = (sim_scores * (sim_ratings - sim_means)).sum()

            denominator = np.absolute(sim_scores).sum()

            weight = (numerator/denominator) if denominator!=0 else 0

            wmean = means[u_idx] + weight

            wmean_rating = wmean[0]

            

    else:

        wmean_rating = 0



    return wmean_rating
# Collaborative filtering with post-filtered contexts

def CaCF_Post(user_id, location_id, s_matrix, c_current, delta):

    

    # Calculate cf

    initial_pred = CF(user_id, location_id, s_matrix)

    

    if location_id in r_df:

        r = np.array(r_df)

        users = r_df.index

        locations = r_df.columns

        l = np.where(locations==location_id)[0]

        c_profile = contexts

        all_cnx = contexts.context.unique().tolist()

        c = np.array(c_profile)

        u_idx = np.where(users==user_id)[0]

        c_current = tuple(c_current)

        

        # Get contexts of similar users visited the location

        l_cnx = np.array(c_profile.loc[c_profile.location_id==location_id,['user_id','context']])

                

        if c_current in all_cnx:

            # Find similarity of the current context to location contexts

            cnx_scores = np.array([[uid, cs_df[c_current][cx]] for uid,cx in l_cnx])



            # Filter users whose similarity bigger than delta

            filtered_scores = cnx_scores[cnx_scores[:,1].astype(float)>delta]



            # Location popularity based on current context

            visit_prob = len(filtered_scores) / len(cnx_scores)

            

        else:

            visit_prob = 1



        return initial_pred * visit_prob



    else:

        return initial_pred
# Find ratings

test_rating = X_test.groupby(['location_id','user_id'])['visit_time'].count().reset_index(name='rating')

test_rating = normalize(test_rating)

r_df_test = test_rating.pivot_table(index='user_id', columns='location_id', values='rating', fill_value=0)



# Proposed approach

def EACOS_CaCF_Post(user_id, location_id, c_current, delta):

    res = CaCF_Post(user_id, location_id, s_df, c_current, delta)

    return res



# Recommendation

def predict(target_user, model, option=None):

    true = r_df_test.loc[target_user]

    

    # Check if model is context-aware 

    if option:

        pred_val = []

        for l in true.index:

            delta = option.get('delta')

            c_current = tuple(X_test.xs(target_user)[['season','daytime','weather']].head(1).values[0])

            r = model(user_id=target_user, location_id=l, c_current=c_current, delta=delta)

            pred_val.append(r)

    else:

        pred_val = [model(user_id=target_user, location_id=l) for l in true.index]



    pred = pd.Series(pred_val, index=true.index)



    return pred
user = '41087279@N00'

options = {

    'delta': .3

}



def item_relevancy(col):

    relevant = 1

    r_color = 'background-color: lime'

    nr_color = 'background-color: red'

    res = []

    for v in col:

        if v > relevant:

            res.append(r_color)

        elif (v > 0) & (v <= relevant):

            res.append(nr_color)

        else:

            res.append('')

    return res

    

true = r_df_test.loc[user]

pred = predict(user, EACOS_CaCF_Post, option=options)



with pd.option_context("display.max_rows", None):

    prediction = pd.DataFrame({'true': true, 'pred': pred})

    display(prediction.style.apply(lambda col: item_relevancy(col)))
# Top 10 recommendations

top_10 = prediction.nlargest(10, 'pred')

top_10.style.apply(lambda col: item_relevancy(col))
def rmse(true, pred):

    return np.sqrt(mean_squared_error(true, pred))



def mean_average_precision(true, pred, k=10):

    relevant = 1

    sort_rates = lambda s: s.sort_values(ascending=False)

    true = [r[1].where(r[1]>relevant).dropna().index.tolist() for r in true.iterrows()]

    pred = [sort_rates(r[1].where(r[1]>relevant).dropna()).index.tolist() for r in pred.iterrows()]

    map_score = mapk(true, pred, k)

    return map_score
def predict_all(model, option=None):

    users = r_df_test.index

    locations = r_df_test.columns

    pred = np.zeros(r_df_test.shape)

    

    for i in range(0,len(users)):

        uid = users[i]

        for j in range(0,len(locations)):

            lid = locations[j]

            # Check if model is context-aware 

            if option:

                delta = option.get('delta')

                c_current = X_test.xs(uid)[['season','daytime','weather']].head(1).values[0]

                pred[i,j] = model(user_id=uid, location_id=lid, c_current=c_current, delta=delta)

            else:

                pred[i,j] = model(user_id=uid, location_id=lid)

                        

    return pd.DataFrame(pred, index=users, columns=locations)
deltas = np.arange(0.1, 1, 0.1)

eval_scores = []



for d in deltas:

    options['delta'] = d

    pred = predict_all(EACOS_CaCF_Post, option=options)

    precision = mean_average_precision(r_df_test,pred)

    eval_scores.append(precision)

    

d_eval = pd.DataFrame(eval_scores, index=deltas, columns=['precision'])



# Delta influence on the prediction and racall

d_precision = go.Figure([go.Scatter(

    name='MAP', 

    x=d_eval.index, 

    y=d_eval.precision, 

    text=d_eval.precision,

    line_shape='spline'

)])



d_precision.update_layout(

    title='The impact of similarity threshold on the recommendation quality',

    xaxis=dict(title='Threshold of context similarity (\u03B4)', autorange='reversed'), 

    yaxis=dict(title='MAP'),

    template=plot_template

)



d_precision.show()
## Non context-aware methodologies with asymetric similarity measure

# Asymmetric cosine similarity

def asymmetric_cosine(m, mf=False, **kwarg):

    # Cosine similarity matrix distance

    cosine = cosine_similarity(m)

    # Asymmetric coefficient

    def asymCo(X,Y):

        co_rated_item = np.intersect1d(np.nonzero(X),np.nonzero(Y)).size

        coeff = co_rated_item / np.count_nonzero(X)

        return coeff

    asym_ind = pairwise_distances(m, metric=asymCo)

    # Sorensen similarity matrix distance

    sorensen = 1 - pairwise_distances(np.array(m, dtype=bool), metric='dice')

    # Final similarity matrix

    usim = np.multiply(np.multiply(cosine,asym_ind),sorensen)

    # Check if matrix factorization was True

    if mf:

        binary = np.invert(usim.astype(bool))*1

        model = NMF(**kwarg)

        W = model.fit_transform(usim)

        H = model.components_

        factorized_usim = np.dot(W,H)*binary + usim

        usim = factorized_usim

            

    return pd.DataFrame(usim, index=m.index, columns=m.index)



# Calculate user similarities

asym_cos = asymmetric_cosine(r_df)

mf_asym_cos = asymmetric_cosine(r_df, mf=True, solver='mu')



# Methods

def ACOS(user_id, location_id):

    res = CF(user_id, location_id, asym_cos)

    return res



def MF_ACOS(user_id, location_id):

    res = CF(user_id, location_id, mf_asym_cos)

    return res
## Context-aware methodologies symmetric similarity measure

# Similarity measure based on location popularity

def loc_pop_sim(df, dist_method='correlation'):

    df = df.reset_index()

    # Calculate location pop

    loc_idf = np.log10(df.groupby('location_id')['user_id'].count().sum()

                    /df.groupby('location_id')['user_id'].count()

                   ).reset_index(name='idf_score')

    loc_idf = df.merge(loc_idf)

    

    # Create location popularity matrix

    r_df = loc_idf.pivot_table(

        index='user_id', 

        columns='location_id', 

        values='idf_score', 

        fill_value=0

    )

    

    # Calculate user similarities

    if dist_method == 'dice':

        dist = 1 - pairwise_distances(r_df.values, metric=dist_method)

    else:

        dist = pairwise_distances(r_df.values, metric=dist_method)

    return pd.DataFrame(dist, r_df.index, r_df.index)



# Calculate user similarities

sym_locpop_pearson = loc_pop_sim(X_train)

sym_locpop_sorensen = loc_pop_sim(X_train, dist_method='dice')



# Methods

def PR(user_id, location_id):

    res = CF(user_id, location_id, sym_locpop_pearson)

    return res



def CSR(user_id, location_id, c_current, delta):

    initial_pred = CF(user_id, location_id, sym_locpop_pearson)

    if location_id in r_df:

        r = np.array(r_df)

        users = r_df.index

        locations = r_df.columns

        l = np.where(locations==location_id)[0]

        c_profile = contexts

        c = np.array(c_profile)

        u_idx = np.where(users==user_id)[0]

        c_current = tuple(c_current)



        # Find users who visit the location in the current context 

        exact_match = contexts[(contexts.location_id==location_id)&(contexts.context==c_current)].user_id.unique()

        

        if exact_match.size != 0:

            idx = np.where(users.isin(exact_match))



            # Calculate visit probability in exact-match context

            visit_match_prob = r[idx,l].sum() / r[:,l].sum()



            # Calculate visit probability of location

            visit_loc_prob = r[:,l].sum() / r.sum()



            # Calculate visit probability in current context

            visit_cnx_prob = contexts[contexts.context==c_current].location_id.count()/r.sum()



            visit_prob = (visit_loc_prob * visit_match_prob) / visit_cnx_prob

        

            return initial_pred * visit_prob

        

        else:

            return initial_pred

    

    else:

        return initial_pred



def Sorensen_CaCF_Post(user_id, location_id, c_current, delta):

    res = CaCF_Post(user_id, location_id, sym_locpop_sorensen, c_current, delta=.3)

    return res
models = [PR, ACOS, MF_ACOS, CSR, Sorensen_CaCF_Post, EACOS_CaCF_Post]

k_range = [5,10,15,20]

eval_scores = {}

true = r_df_test

options['delta'] = .3



for model in models:

    option = None if model.__name__ in ['ACOS', 'PR', 'MF_ACOS'] else options

    val = []

    for k in k_range:

        pred = predict_all(model, option)

        mapk_score = mean_average_precision(true, pred, k)

        val.append(mapk_score)

        

    eval_scores[model.__name__] = val

    

map_at_k = pd.DataFrame(eval_scores, index=k_range)



mapk_comp = go.Figure()



for model, ser in map_at_k.iteritems():

    mapk_comp.add_trace(go.Bar(

        name=model,

        x=ser.index,

        y=ser.values,

        width=.6

    ))

    

mapk_comp.update_layout(

    barmode='group',

    title='Comparision of the proposed method with the benchmarking methods (MAP@k)',

    xaxis=dict(title='NUmber of recommendations'),

    yaxis=dict(title='MAP@k', range=[.7,.9]),

    template=plot_template

)



mapk_comp.show()
rmse_eval = []



for model in models:

    option = None if model.__name__ in ['ACOS', 'PR', 'MF_ACOS'] else options

    pred = predict_all(model, option)

    rmse_score = rmse(true, pred)

    rmse_eval.append([model.__name__, rmse_score])

    

rmse_perf = pd.DataFrame(rmse_eval, columns=['model','value'])



rmse_comp = go.Figure([go.Bar(

    x=rmse_perf.model, 

    y=rmse_perf.value,

    width=.5,

    text=round(rmse_perf.value,2),

    textposition='outside', 

    marker=dict(color=rmse_perf.index, colorscale='Viridis')

)])



rmse_comp.update_layout(

    barmode='group',

    title='Comparision of the proposed method with the benchmarking methods (RMSE)',

    yaxis=dict(title='RMSE'),

    template=plot_template

)



rmse_comp.show()