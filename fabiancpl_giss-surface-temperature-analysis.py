import numpy as np
from collections import OrderedDict

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import *
%matplotlib inline

from ipywidgets import interact, interactive, IntSlider, Layout, interact_manual
import ipywidgets as widgets
from IPython.display import display

from sklearn.linear_model import LinearRegression
# Global Temperature
dts_global_df = pd.read_csv( '../input/GLB.Ts.csv', na_values = [ '***' ] )
# Temperature by Hemisphere
dts_northern_df = pd.read_csv( '../input/NH.Ts.csv', na_values = [ '***' ] )
dts_southern_df = pd.read_csv( '../input/SH.Ts.csv', na_values = [ '***' ] )
nino_nina_df = pd.read_csv( '../input/nino-nina-periods.csv' )
dts_global_df = dts_global_df[ [ dts_global_df.columns[ 0 ] ] + dts_global_df.columns[ -4: ].tolist() ]
dts_northern_df = dts_northern_df[ [ dts_northern_df.columns[ 0 ] ] + dts_northern_df.columns[ -4: ].tolist() ]
dts_southern_df = dts_southern_df[ [ dts_southern_df.columns[ 0 ] ] + dts_southern_df.columns[ -4: ].tolist() ]
dts_global_df = dts_global_df.loc[ dts_global_df[ 'Year' ] >= 2002 ]
dts_northern_df = dts_northern_df.loc[ dts_northern_df[ 'Year' ] >= 2002 ]
dts_southern_df = dts_southern_df.loc[ dts_southern_df[ 'Year' ] >= 2002 ]
dts_global_df = dts_global_df.set_index( 'Year' ).unstack().reset_index()[ [ 'Year', 'level_0', 0 ] ] \
    .sort_values( by = 'Year' ).rename( columns = { 'level_0' : 'Season', 0 : 'Temperature' } )

dts_northern_df = dts_northern_df.set_index( 'Year' ).unstack().reset_index()[ [ 'Year', 'level_0', 0 ] ] \
    .sort_values( by = 'Year' ).rename( columns = { 'level_0' : 'Season', 0 : 'Temperature' } )

dts_southern_df = dts_southern_df.set_index( 'Year' ).unstack().reset_index()[ [ 'Year', 'level_0', 0 ] ] \
    .sort_values( by = 'Year' ).rename( columns = { 'level_0' : 'Season', 0 : 'Temperature' } )
dts_global_df.loc[ dts_global_df[ 'Season' ] == 'DJF', 'Season' ] = 1
dts_global_df.loc[ dts_global_df[ 'Season' ] == 'MAM', 'Season' ] = 2
dts_global_df.loc[ dts_global_df[ 'Season' ] == 'JJA', 'Season' ] = 3
dts_global_df.loc[ dts_global_df[ 'Season' ] == 'SON', 'Season' ] = 4

dts_northern_df.loc[ dts_northern_df[ 'Season' ] == 'DJF', 'Season' ] = 1
dts_northern_df.loc[ dts_northern_df[ 'Season' ] == 'MAM', 'Season' ] = 2
dts_northern_df.loc[ dts_northern_df[ 'Season' ] == 'JJA', 'Season' ] = 3
dts_northern_df.loc[ dts_northern_df[ 'Season' ] == 'SON', 'Season' ] = 4

dts_southern_df.loc[ dts_southern_df[ 'Season' ] == 'DJF', 'Season' ] = 1
dts_southern_df.loc[ dts_southern_df[ 'Season' ] == 'MAM', 'Season' ] = 2
dts_southern_df.loc[ dts_southern_df[ 'Season' ] == 'JJA', 'Season' ] = 3
dts_southern_df.loc[ dts_southern_df[ 'Season' ] == 'SON', 'Season' ] = 4
dts_global_df.sort_values( by = [ 'Year', 'Season' ], inplace = True )
dts_northern_df.sort_values( by = [ 'Year', 'Season' ], inplace = True )
dts_southern_df.sort_values( by = [ 'Year', 'Season' ], inplace = True )
dts_global_df[ 'Year-Season' ] = dts_global_df[ 'Year' ].astype( str ) + '-' + dts_global_df[ 'Season' ].astype( str )
dts_northern_df[ 'Year-Season' ] = dts_northern_df[ 'Year' ].astype( str ) + '-' + dts_northern_df[ 'Season' ].astype( str )
dts_southern_df[ 'Year-Season' ] = dts_southern_df[ 'Year' ].astype( str ) + '-' + dts_southern_df[ 'Season' ].astype( str )
dts_global_df.tail()
nino_nina_df.tail()
dts_global_df = dts_global_df.reset_index()
del dts_global_df[ 'index' ]
global_model = LinearRegression( normalize = True )
X = np.reshape( dts_global_df.index, ( len( dts_global_df.index ), 1 ) )
global_model.fit( X[ :-1 ], dts_global_df[ 'Temperature' ][ :-1 ] ) 
dts_global_df[ 'Trend' ] = global_model.predict( X )
dts_northern_df = dts_northern_df.reset_index()
del dts_northern_df[ 'index' ]
northern_model = LinearRegression( normalize = True )
X = np.reshape( dts_northern_df.index, ( len( dts_northern_df.index ), 1 ) )
northern_model.fit( X[ :-1 ], dts_northern_df[ 'Temperature' ][ :-1 ] ) 
dts_northern_df[ 'Trend' ] = northern_model.predict( X )
dts_southern_df = dts_southern_df.reset_index()
del dts_southern_df[ 'index' ]
southern_model = LinearRegression( normalize = True )
X = np.reshape( dts_southern_df.index, ( len( dts_southern_df.index ), 1 ) )
southern_model.fit( X[ :-1 ], dts_southern_df[ 'Temperature' ][ :-1 ] ) 
dts_southern_df[ 'Trend' ] = southern_model.predict( X )
# https://matplotlib.org/examples/color/colormaps_reference.html
def get_color( i ):
    cmap = plt.get_cmap( 'RdBu' )
    return rgb2hex( cmap( i )[ :3 ] )
c_nino = { 'Slight': get_color( .4 ), 'Medium' : get_color( .3 ), 'Strong' : get_color( .2 ), 'Meganiño' : get_color( .1 ) }
c_nina = { 'Slight': get_color( .6 ), 'Medium' : get_color( .7 ), 'Strong' : get_color( .8 ), 'Meganiña' : get_color( .9 ) }
def show_graph( global_selected, northern_selected, southern_selected ):
    
    plt.figure( figsize = ( 20, 7 ) )
    
    if global_selected:
        plt.plot( dts_global_df[ 'Year-Season' ], dts_global_df[ 'Temperature' ], color = 'green', label = 'Global' )
        plt.plot( dts_global_df[ 'Year-Season' ], dts_global_df[ 'Trend' ], color = 'green', alpha = 0.5, linestyle = '--', label = '' )

    if northern_selected:
        plt.plot( dts_northern_df[ 'Year-Season' ], dts_northern_df[ 'Temperature' ], color = 'orange', label = 'Northern Hemisphere' )
        plt.plot( dts_northern_df[ 'Year-Season' ], dts_northern_df[ 'Trend' ], color = 'orange', alpha = .8, linestyle = '--', label = '' )
    
    if southern_selected:
        plt.plot( dts_southern_df[ 'Year-Season' ], dts_southern_df[ 'Temperature' ], color = 'purple', label = 'Southern Hemisphere' )
        plt.plot( dts_southern_df[ 'Year-Season' ], dts_southern_df[ 'Trend' ], color = 'purple', alpha = .8, linestyle = '--', label = '' )
    
    plt.axhline( y = 0, linewidth = 1, color = 'gray', linestyle = '--' )
    plt.axhline( y = 0, linewidth = 1, color = 'gray', linestyle = '--' )
    plt.title( 'Global Temperature Anomalies - Meteorological Station Data' )
    plt.xticks( rotation = 'vertical' )
    plt.xlabel( 'Year - Season' )
    plt.ylabel( 'Temperature Anomaly Mean' )
    plt.grid( linestyle = ':', linewidth = .5 )

    for index, row in nino_nina_df.loc[ nino_nina_df[ 'Event' ] == 'El Niño' ].iterrows():
        plt.axvspan( row[ 'Period-Min' ], row[ 'Period-Max' ], facecolor = c_nino[ row[ 'Level' ] ], alpha = .8, label = 'El Niño - ' + row[ 'Level' ] )

    for index, row in nino_nina_df.loc[ nino_nina_df[ 'Event' ] == 'La Niña' ].iterrows():
        plt.axvspan( row[ 'Period-Min' ], row[ 'Period-Max' ], facecolor = c_nina[ row[ 'Level' ] ], alpha = .8, label = 'La Niña - ' + row[ 'Level' ] )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict( zip( labels, handles ) )
    plt.legend( by_label.values(), by_label.keys() )

    plt.show()
    
    return None
p = interactive( show_graph, global_selected = True, northern_selected = False, southern_selected = False )
display( p )

