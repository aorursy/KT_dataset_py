import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from math import pi

db_df=pd.read_csv("../input/database.csv")
print( "Read %d records" % (len( db_df )))

db_df.head( 3 )

allDomains = sorted( db_df[ 'domain' ].unique().tolist() )
print( "All domains: %s" % allDomains )

# Cut dataset down to only what's needed
db_df = db_df[ ['birth_year', 'domain'] ]

# The birth year column contains some strings, e.g. 'Unknown'.
# Convert column to numeric value, the strings will be set to NaN
db_df[ 'birth_year_num' ] = pd.to_numeric( db_df['birth_year'], errors='coerce' )

# Remove NaN rows
db_df = db_df[ db_df[ 'birth_year_num' ].notnull() ]

db_df.head( 3 )

  
yearStep = 25
startYear = 1900
endYear = 2000

allCounts = []
allTitles = []
maxCount = 0

db_df.head( 3 )

# Slice up dataset by quarters of 20th Century
for year in range( startYear, endYear, yearStep):

    # Get rows by birth year bracket
    quarter_df = db_df.loc[ (db_df['birth_year_num'] >= year) & (db_df['birth_year_num'] < (year + 25))]
    
    print( "Year %d Row count: %d" % (year, len( quarter_df )))

    # Get domain counts, sorted by domain name
    counts_s = quarter_df[ 'domain' ].value_counts()
    counts_s = counts_s.sort_index()

    # Global maximum
    maxCount = max( maxCount, counts_s.max() )
    
    # Domain may be missing so fill in count list 
    domainCounts = []
    for domain in allDomains:
        
        if counts_s.index.contains( domain ):            
            domainCounts.append( counts_s[ domain ] )
        else:
            domainCounts.append( 0 )

    domainCounts += domainCounts[:1]
    print( "Counts: ", domainCounts )

    allCounts.append( domainCounts )   
    allTitles.append( "%d - %d" % (year, year + yearStep - 1) )

domainCount = len( allDomains )
angles = [n / float( domainCount ) * 2 * pi for n in range( domainCount )]
angles += angles[:1]

# Make figure biggishly
plt.figure(figsize=(10,10), dpi=96 ) 

facetCount = 1
print( allTitles )
# Build axis for each set of counts    
for counts in allCounts:    

    # Set up Axis
    ax = plt.subplot( 2, 2, facetCount, polar=True, )
    
    ax.set_title( allTitles.pop( 0 ), loc='left', fontweight='bold' )
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction( -1 )
        
    # Draw ylabels
    ax.set_rlabel_position( 0 )
    
    facetCount += 1
    
    ax.plot( angles, counts, linewidth=1, linestyle='solid', label=(str(year)), color='b' )
    ax.fill( angles, counts, 'b', alpha=0.2 )
    plt.xticks( angles[:-1], allDomains, color='grey' )
    plt.yticks( color='grey')
    plt.ylim( 0, maxCount )

plt.subplots_adjust(wspace=0.75, hspace=0.25 ) 
plt.suptitle('Domain Counts by Birth Year', fontsize=16, fontweight='bold' )