import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm
from collections import Counter
from time import time
from matplotlib import collections  as mc

sns.set_style('whitegrid')
# Get all the cities:   Note: CityId is equal to the index
df_cities = pd.read_csv("../input/traveling-santa-2018-prime-paths/cities.csv")
##df_cities.head(10)
def sieve_eratosthenes(n):
    prime_flags = [False, False] + [True for i in range(n-1)]
    p = 2
    while (p * p <= n):
        if (prime_flags[p] == True):
            for i in range(p * 2, n + 1, p):
                prime_flags[i] = False
        p += 1
    return prime_flags
max_city = max(df_cities.CityId)
print("Largest City number to visit : ", max_city)
prime_flags = np.array(sieve_eratosthenes(max_city)).astype(int)
df_cities['Prime'] = prime_flags

df_cities.head(5)
df_cities.tail()
# Assign a distance measure from xc,yc to all cities
# If remaining is present then only that subset are calculated (if few enough)
# and/but the return array is full size.
def dist_array(coords_in, xc, yc, remaining=[]):
    begin = np.array([xc, yc])[:, np.newaxis]
    # Doing all cities is generally faster, unless a small number (< 60000)
    if (len(remaining) < 1) or (len(remaining) > 60000):
        dists = np.linalg.norm(coords_in - begin, ord=2, axis=0)
    else:
        dists = np.zeros((len(coords_in[0])))
        dists[remaining] = np.linalg.norm(coords_in[: , remaining] - begin, ord=2, axis=0)
    return dists

# return the index of the nearest available city, and its distance
def get_next_city(dist, avail):
    min_avail = np.argmin(dist[avail])
    return avail[min_avail], dist[avail[min_avail]]

def plot_path(path, coordinates, np_star=True, end_stars=False, prime_vals=[]):
    # Plot tour
    lines = [[coordinates[: ,path[i-1]], coordinates[:, path[i]]] for i in range(1, len(path))]
    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = plt.subplots(figsize=(15,10))
    ax.set_aspect('equal')
    plt.grid(False)
    ax.add_collection(lc)
    ax.autoscale()
    if np_star:
        # add the North Pole location
        plt.scatter(coordinates[0][0], coordinates[1][0], s=150, c="red", marker="*", linewidth=3)
        # and first cities on the path
        ##plt.scatter(coordinates[0][path[1:10]], coordinates[1][path[1:10]], s=15, c="black")
    if end_stars:
        # add stars at first and last location
        plt.scatter(coordinates[0][path[0]], coordinates[1][path[0]], s=150, c="red", marker="*", linewidth=3)
        plt.scatter(coordinates[0][path[-1]], coordinates[1][path[-1]], s=150, c="red", marker="*", linewidth=3)
    if len(prime_vals) == len(coordinates[0]):
        # Mark the prime cities
        for this_city in path:
            if prime_vals[this_city] == 1:
                plt.scatter(coordinates[0][this_city], coordinates[1][this_city],
                            s=60, c="green", marker="*", linewidth=3)
    plt.show()
    
# Calculate the Score, Carrots, and Length for a path
def usual_score(path, coords, prime_flags):
    score = 0
    carrots = 0 
    length = 0
    for i in range(1, len(path)):
        begin = path[i-1]
        end = path[i]
        distance = np.linalg.norm(coords[:, end] - coords[:, begin], ord=2)
        length += distance
        if i % 10 == 0:
            # if the starting city is prime then a carrot and no penalties
            if prime_flags[begin]:
                carrots += 1
            # if not prime, no carrot and a penalty
            else:
                distance *= 1.1
        score += distance

    return score, carrots, length

def str_from_cities(cities_list):
    # Create a comma-separated string of cities_list
    cities_str = ''
    for icty in range(len(cities_list)):
        cities_str += str(int(cities_list[icty]))+', '
    return cities_str

def cities_from_str(cities_str):
    # extract a list of city ids from cities_str,
    pieces = cities_str.split(",")
    # start with an empty list
    city_list = []
    # if the first city has value >= 0 then get and return the cities
    if int(pieces[0]) >= 0:
        for ipiece in range(len(pieces)-1):
            city_list.append(int(pieces[ipiece]))
    return city_list

def add_to_segment(bb_next, n_between, mpdist_limit=1.e6):        
    # Routine to increase selected segment, bb_next, to have n_between cities,
    # will add/extend to have n_between nearest-to-mid-point cities in the path.
    # Will not add cities that are further than mpdist_limit.
    # If the segment is partially filled, the centroid of cities replaces backbone mid-point.
    #
    # Use: n_added = add_to_segment(1,8)  # add 8 from NP to first bb city
    
    # Assumes access to read these variables:
    #  backbone[]   coordinates[[][]]
    #
    # Set these to global to allow modifying values as well:
    global dfbb
    global remaining

    # Return the number that were added
    n_added = 0
    
    # Don't do bb 0:
    if bb_next == 0:
        return n_added
    
    # Check how many are in this segment already, return if there are enough already.
    seg_cities = cities_from_str(dfbb.loc[bb_next,'CityListStr'])
    if len(seg_cities) >= n_between:
        print("Segment already full for bb_next =", backbone[bb_next])
        return n_added
    
    # Need to add one or more, so ...
    # If this backbone segment is empty then start it with one city
    if len(seg_cities) == 0:
        # Calc midpoint (average location) between bb_last and bb_next (faster than getting it from the dfbb?)
        mid_pt = 0.5*(coordinates[:, backbone[bb_next - 1]] + coordinates[:, backbone[bb_next]])
        # Calc distances of all remaining points from this mid_pt
        dist_from_mid = dist_array(coordinates, mid_pt[0], mid_pt[1], remaining=remaining)
        
        # Start with the remaining city closest to midpoint
        add_this, test_dist = get_next_city(dist_from_mid, remaining)
        # Only start it if its within mpdist_limit:
        if test_dist < mpdist_limit:
            this_dist = 1.0*test_dist
            remaining = np.setdiff1d(remaining, add_this)
            n_added += 1
            # and create a path with that city in the middle
            built_path = [backbone[bb_next - 1], add_this, backbone[bb_next]]
            _dum1, _dum2, this_len = usual_score(built_path, coordinates, prime_flags)
            best_score = this_len
        else:
            # Too far away, forget it
            # Nothing to update
            return n_added
    else:
        # Assemble the already assigned cities into built_path with backbone endpoints
        built_path = [backbone[bb_next - 1]] + seg_cities + [backbone[bb_next]]
        #
        # Calc the centroid of all cities on this path (use same mid_pt variable)
        mid_pt = 1.0*coordinates[:, built_path[0]]
        for seg_city in built_path[1:]:
            mid_pt += coordinates[:, seg_city]
        mid_pt = mid_pt/len(built_path)
        # Calc distances of all remaining points from this mid_pt
        dist_from_mid = dist_array(coordinates, mid_pt[0], mid_pt[1], remaining=remaining)
        
    # Now, add additional cities... Add them in order from closest to farthest from midpoint
    # and/but choose where in the segment path is best when inserting each city.
    for iadd in range(len(built_path)-1,n_between+1):
        add_this, test_dist = get_next_city(dist_from_mid, remaining)
        # Only add it if its within mpdist_limit:
        if test_dist < mpdist_limit:
            this_dist = 1.0*test_dist
            remaining = np.setdiff1d(remaining, add_this)
            n_added += 1
            # Loop over insertion location
            # keep track of shortest path and where the insertion for it was
            best_score = 1.e9
            best_insert_path = []
            for iinsert in range(1, len(built_path)):
                test_path = built_path[:iinsert] + [add_this]+ built_path[iinsert:]
                _dum1, _dum2, this_len = usual_score(test_path, coordinates, prime_flags)
                # check the length for this path
                if this_len < best_score:
                    best_score = this_len
                    best_insert_path = test_path
            # OK, put the added city in best location in the path:
            built_path = best_insert_path
        else:
            # too far away, stop adding here
            break  # out of the for loop adding cities
        # Go back to get next city for the segment
        
    # Done getting and arranging the desired number of cities for the segment
    # 
    if n_added > 0:
        # save these cities (without bb end points) in the dfbb
        dfbb.loc[bb_next, 'CityListStr'] = str_from_cities(built_path[1 : -1])
        # save the number of cities assigned to the segment
        dfbb.loc[bb_next, 'NumCities'] = len(built_path[1 : -1])
        # and the (max) distance of the last selected city
        dfbb.loc[bb_next, 'MaxRadius'] = this_dist
        # and the current length of the backbone segment
        dfbb.loc[bb_next, 'Length'] = best_score
    
    # end of add_to_segment()
    return n_added
# Get the Backbone path

# TSP solution for the North Pole and the Prime cities:
##df_bbtsp = pd.read_csv("../input/santa-18-primes-tsp60/primes_path_t60.csv")
# Slightly improved all primes TSP
df_bbtsp = pd.read_csv("../input/santa-18-primes-tsp60/primes_path_t5000.csv")
bb_name = "bb_primesTSP5000"

df_bbtsp.head()
df_bbtsp.tail()
# Generate the backbone list of CityIds, starting and ending at NP
# Note that the 'Path' column values are prime-city-indices in path order,
# that is, the loc of the CityId in the df.
backbone = list(df_bbtsp.loc[list(df_bbtsp['Path']),'CityId'])

# loop back to the start of the bb (e.g. the NP for a usual bb)
backbone += [backbone[0]]
len_bb = len(backbone)
print("Backbone length is", len_bb, "\nStarts with cities:\n  ", backbone[0:10],
     "\nEnds with cities:\n    ", backbone[len_bb-10:])
# Show all the cities and the backbone cities
plt.figure(figsize=(15, 10))
# All cities:
plt.scatter(df_cities.X, df_cities.Y, s=1, c="green", alpha=0.3)
# Prime cities:
plt.scatter(df_cities.loc[backbone, "X"], df_cities.loc[backbone, "Y"], s=1, c="red")
plt.scatter(df_cities.iloc[0: 1, 1], df_cities.iloc[0: 1, 2], s=30, c="black")
plt.grid(False)
plt.show()
# Put all the city coordinates in an np array
coordinates = np.array([df_cities.X, df_cities.Y])

# Look at the backbone (red star is NP)
# whole thing
plot_path(backbone, coordinates)
# Can calculate the path-length of the backbone-path (NP to NP) itself
_dum1, _dum2, bb_len = usual_score(backbone+[0], coordinates, prime_flags)

print("Backbone path-length is ", int(bb_len))
# * * * Can use pre-made file * * *
# Read in previously saved backbone dataframe:
# Full backbone with Primes and non-primes
##dfbb = pd.read_csv("dfbb_primesTSPremaining_v6_SAVE.csv", sep=";", index_col=0)
# Backbone of just the Primes
##dfbb = pd.read_csv("dfbb_primesTSP5000_v8_empty.csv", sep=";", index_col=0)
# * * to Skip the following cells * * * *
#  Or, make and fill the dfbb...
# Create a dataframe to keep track of information about the various segments of the path
#(segment = piece = cities between adjacent bb cities = cities on path between this and previous bb city).
dfbb = df_cities.loc[backbone].copy().reset_index()

# drop the new "index" column (same as CityId)
dfbb = dfbb.drop(columns='index')

dfbb.head()
# Create a dataframe to keep track of information about the various segments of the path
#(segment = piece = cities between adjacent bb cities = cities on path between this and previous bb city).
dfbb = df_cities.loc[backbone].copy().reset_index()

# drop the "index" column
dfbb = dfbb.drop(columns='index')

# Coordinates of the segment midpoint
dfbb["MidX"] = 0.0
dfbb["MidY"] = 0.0

# The direct length from previous bb city to this bb city
dfbb['BBtoBBlen'] = 0.0

# Define these other values in dfbb - they give information on the solution status
# The maximum radial distance from the segment midpoint of the cities assigned to this segment
dfbb['MaxRadius'] = 0.0
# Define and initialize "RadAlone" in dataframe (min midpoint radius with 9 cities)
dfbb['RadAlone'] = 0.0
# The length of the segment as currently constructed
dfbb['Length'] = 0.0
# Define and initialize "LenAlone" in dataframe (est. of min length with 9 closest cities)
dfbb['LenAlone'] = 0.0
# The number of cities assigned to this segment (e.g. in CityListStr)
dfbb['NumCities'] = 0
# A string entry to keep track of the cities that are assigned to this segment
dfbb['CityListStr'] = '-1,'

# Fill the BBtoBBlen and Mid point values (they don't change)
for bb_next in range(1,len(backbone)):
    xy_next = coordinates[:, backbone[bb_next]]
    xy_last = coordinates[:, backbone[bb_next-1]]
    dfbb.loc[bb_next, 'BBtoBBlen'] = np.linalg.norm(xy_next - xy_last, ord=2)
    mid_pt = 0.5*(xy_last + xy_next)
    dfbb.loc[bb_next, "MidX"] = mid_pt[0]
    dfbb.loc[bb_next, "MidY"] = mid_pt[1]
    if bb_next % 1000 == 0:
        print(" ... ", bb_next, "bb lengths, mid-points calculated.")

# Keep track of time...
t0 = time()

# Start with all cities:
remaining = np.array(df_cities.CityId)
# Remove the ones in the backbone
all_remaining = np.setdiff1d(remaining, np.array(backbone))

# First segment (special)
bb_next = 1
remaining = all_remaining.copy()
n_added = add_to_segment(bb_next, 8)

# all the rest
for bb_next in range(2,len(backbone)):
    remaining = all_remaining.copy()
    n_added = add_to_segment(bb_next, 9)
    # show the progress
    if bb_next % 500 == 0:
        print( int(time() - t0), "seconds   - ...added bb number", bb_next)
        
# This takes longer because there are the maximum remaining number of cities for every segment.
# Could imagine writing out the dfbb dataframe at this point,
# its values are useful to a particular backbone.
# Transfer/Save these values
dfbb['LenAlone'] = dfbb['Length']
dfbb['RadAlone'] = dfbb['MaxRadius']
# and reset these values
dfbb['MaxRadius'] = 0.0
dfbb['Length'] = 0.0
dfbb['CityListStr'] = '-1,'
# Save the dataframe since it takes a while to create...
# Use ";" as separation character since I use commas in CityListStr.
dfbb.to_csv("df"+bb_name+"_v8_empty.csv", sep=";")
# * * * Below assumes dfbb was either made (as above) or read in (farther above) * * *
# Look at the dfbb
dfbb.head(5)
dfbb.tail(5)
# The total LenAlone, roughly the (optimistic?) best one could get with this backbone and these cities.
dfbb['LenAlone'].sum()
# Show the "alone" properties of the segments
pd.plotting.scatter_matrix(dfbb[['BBtoBBlen','RadAlone','LenAlone']], figsize=(14,9),
                           diagonal='hist', range_padding=0.05, hist_kwds={'bins':100},
                           grid=True, alpha=0.5)
plt.show()
# (Re)Start by resetting the city-selection values in dfbb:

def reset_segments():
    global bdfbb, remaining
    
    # The maximum radial distance from the segment midpoint of the cities assigned to this segment
    dfbb['MaxRadius'] = 0.0
    # The length of the segment as currently constructed
    dfbb['Length'] = 0.0
    # The number of cities assigned to this segment (e.g. in CityListStr)
    dfbb['NumCities'] = 0
    # A string entry to keep track of the cities that are assigned to this segment
    dfbb['CityListStr'] = '-1,'

    # Initialize the "remaining" cities to be the ones not in the backbone
    # Start with all cities:
    remaining = np.array(df_cities.CityId)
    # Remove the ones in the backbone
    remaining = np.setdiff1d(remaining, np.array(backbone))

    print("Total number of cities (includes NP):", max_city+1,
          "\nUnique cities in the backbone:", len_bb-1,
          "\nNumber of cities not in the backbone: ", len(remaining),
          "\n   Total primes among these:", sum(prime_flags[remaining]),  "<-- should be zero!")

# And do it
reset_segments()

# Calculate how many cities will be between the last non-NP bb city and the (return to the) NP 
# so that there will be a multiple of 10 remaining after adding between the bb cities.
n_last_bb_to_NP = ((max_city+1) - 10*(len_bb-2)) % 10
print("\nPut", n_last_bb_to_NP, "cities in the final stretch back to the NP.")
# Generate a path that has 9 non-backbone cities put before/between each backbone city;
# except: 8 between NP and first bb city,
# and a calculated number, n_last_bb_to_NP, between next-to-last bb city and NP.
#

# Keep track of the clock time
t0 = time()

# number per backbone segment, 9 unless doing an experiment
n_per_seg = 9


# Fill the first and last segment
if True:
    # First do NP to first city (special number between):
    n_added = add_to_segment(1, 8)

    # Do the last bb city back to NP (could be a special number, it's 9 actually):
    n_added = add_to_segment(len(backbone)-1, n_last_bb_to_NP)


# Do the "Compact" Primes
# - Even and then Odd bb index ones ("every other"-ish spacing)
# - Select the 9 in sets: first 4, then 3 more, then last 2
# - Centroid used for distance determination
# - No distance limit in adding cities
if True:

    # - - - "compact" ones have the RadAlone/BBtoBBlen ratio less than a value
    compact_ratio = 1.60
    # Do ones that are 'compact':
    ibb_compact = dfbb[(dfbb['RadAlone'] < compact_ratio*dfbb['BBtoBBlen']) & (dfbb['NumCities'] < n_per_seg) & \
                  (dfbb.index % 2 == 0) & (dfbb.index > 1) & (dfbb['Prime'] == 1)].index
    print("\nDoing", len(ibb_compact),"Compact Even Prime cities...")
    ndone = 0
    for bb_next in ibb_compact:
        # Do all 9 at once...
        ##n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=1.e6)
        # Do some, then some more, then do the remaining ones (uses centroid of the some+2)
        n_added = add_to_segment(bb_next, 4, mpdist_limit=1.e6)
        n_added = add_to_segment(bb_next, 7, mpdist_limit=1.e6)
        n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=1.e6)
        #
        ndone += 1
        if ndone % 500 == 0:
            print( int(time() - t0), "seconds   - ...added bb number", bb_next)
            print("            Number remaining:", len(remaining))

    ibb_compact = dfbb[(dfbb['RadAlone'] < compact_ratio*dfbb['BBtoBBlen']) & (dfbb['NumCities'] < n_per_seg) & \
                  (dfbb.index % 2 == 1) & (dfbb.index > 1) & (dfbb['Prime'] == 1)].index
    print("\nDoing", len(ibb_compact),"Compact Odd Prime cities...")
    ndone = 0
    for bb_next in ibb_compact:
        # Do all 9 at once...
        ##n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=1.e6)
        # Do some, then some more, then do the remaining ones (uses centroid of the some+2)
        n_added = add_to_segment(bb_next, 4, mpdist_limit=1.e6)
        n_added = add_to_segment(bb_next, 7, mpdist_limit=1.e6)
        n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=1.e6)
        #
        ndone += 1
        if ndone % 500 == 0:
            print( int(time() - t0), "seconds   - ...added bb number", bb_next)
            print("            Number remaining:", len(remaining))


mpdist_limit = 15.0     
while mpdist_limit < 400.0:

    print("\n\n--- Doing dist limit of ", mpdist_limit, "---")
    # Do the un-finished ODD and EVEN PRIMES
    
    # Do the  - EVEN ones -
    ibb_even = dfbb[(dfbb.index % 2 == 0) & (dfbb['NumCities'] < n_per_seg) & \
               (dfbb.index > 1) & (dfbb['Prime'] == 1)].index
    print("\nDoing", len(ibb_even),"Even Prime cities...")
    ndone = 0
    for bb_next in ibb_even:
        n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=mpdist_limit)
        ndone += 1
        if ndone % 500 == 0:
            print( int(time() - t0), "seconds   - ...added bb number", bb_next)
            print("            Number remaining:", len(remaining))
            
    # Do the  - ODD ones ( > 1 ) -
    ibb_odd = dfbb[(dfbb.index % 2 == 1) & (dfbb.index > 1) & (dfbb['NumCities'] < n_per_seg) & \
              (dfbb.index > 1) & (dfbb['Prime'] == 1)].index
    print("\nDoing", len(ibb_odd),"Odd Prime cities...")
    ndone = 0
    for bb_next in ibb_odd:
        n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=mpdist_limit)
        ndone += 1
        if ndone % 500 == 0:
            print( int(time() - t0), "seconds   - ...added bb number", bb_next)
            print("            Number remaining:", len(remaining))

    mpdist_limit += 5.0 + 0.1*mpdist_limit

    
if True:
    # All/Any remaining cities added
    # distance limit (1.e6 if none)
    mpdist_limit = 1.e6
    
    # Do the un-finished ODD and EVEN PRIMES
    
    # Do the  - EVEN ones -
    ibb_even = dfbb[(dfbb.index % 2 == 0) & (dfbb['NumCities'] < n_per_seg) & \
               (dfbb.index > 1) & (dfbb['Prime'] == 1)].index
    print("\nDoing", len(ibb_even),"Even Prime cities...")
    ndone = 0
    for bb_next in ibb_even:
        n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=mpdist_limit)
        ndone += 1
        if ndone % 500 == 0:
            print( int(time() - t0), "seconds   - ...added bb number", bb_next)
            print("            Number remaining:", len(remaining))
            
    # Do the  - ODD ones ( > 1 ) -
    ibb_odd = dfbb[(dfbb.index % 2 == 1) & (dfbb.index > 1) & (dfbb['NumCities'] < n_per_seg) & \
              (dfbb.index > 1) & (dfbb['Prime'] == 1)].index
    print("\nDoing", len(ibb_odd),"Odd Prime cities...")
    ndone = 0
    for bb_next in ibb_odd:
        n_added = add_to_segment(bb_next, n_per_seg, mpdist_limit=mpdist_limit)
        ndone += 1
        if ndone % 500 == 0:
            print( int(time() - t0), "seconds   - ...added bb number", bb_next)
            print("            Number remaining:", len(remaining))

            
# All done selecting cities within each backbone segment
print( int(time() - t0), "seconds   - Finished to bb number", bb_next)
print("            Number remaining:", len(remaining))

print(f"Loop lasted {(time() - t0) // 60} minutes ")

# Show the dfbb
dfbb.head(5)
dfbb.tail(5)
# Assemble the path from the dfbb
# Start with zeroth bb (NP usually)
path_df = [backbone[0]]
# add the rest
for ibb in range(1,len(backbone)):
    path_df += cities_from_str(dfbb.loc[ibb,'CityListStr'])
    path_df.append(backbone[ibb])
##path_df
# Show the whole path
plot_path(path_df, coordinates)
print("Number of cities in the path:", len(path_df), ";    Number left to add:", len(remaining))
# Show the Rudolph Score results  *** Got All the Carrots ! ***
score, carrots, length = usual_score(path_df, coordinates, prime_flags)
# and without going back to the NP
score_noNP, dummy1, dummy2 = usual_score(path_df[:-1], coordinates, prime_flags)

print("Rudolph Score:", int(score), "   Carrots:", carrots, "   Length:", int(length), ".\n" +
      " Penalty frac:", int(10000*(score-length)/length)/100,
      "%   Final step to NP has distance ", int(score - score_noNP))

# The length from the dataframe values
dfbb['Length'].sum()
# Save the dataframe with the assigned cities.
# Use ";" as separation character since I use commas in CityListStr.
dfbb.to_csv("df"+bb_name+"_v8x_filled.csv", sep=";")
# Look at particular segment of the bb, bb_last to bb_next
if True:
    bb_next = 106

    
    # For testing, etc.
    # Start with empty segment for testing...
    ##reset_segments()
    ##print("")
    #
    # For testing, add to the segment
    ##print("Added", add_to_segment(bb_next, 2, mpdist_limit=1.e6), " Remaining cities:", len(remaining))
    #   Segment from,to : 196117 88513 
    #        has path:  [142958, 80942]
    #   Segment from,to : 196117 88513 
    #      has path:  [85808, 142958, 80942, 81345]
    ##print("")
    ##print("Added", add_to_segment(bb_next, 4, mpdist_limit=1.e6), " Remaining cities:", len(remaining))
    
    seg_cities = cities_from_str(dfbb.loc[bb_next,'CityListStr'])
    print("Segment from,to :", backbone[bb_next - 1], backbone[bb_next], 
         "\n   has path: ", seg_cities)

    segment_path = [backbone[bb_next - 1]] + seg_cities + [backbone[bb_next]]
plot_path(segment_path, coordinates, np_star=False, end_stars=True, prime_vals=prime_flags)
# Total length of these:
seg_len = 0.0
for iseg in range(1,len(segment_path)):
    seg_len += np.linalg.norm(df_cities.loc[segment_path[iseg-1]][['X','Y']] - 
                              df_cities.loc[segment_path[iseg]][['X','Y']], ord=2)
seg_len
# closeup of the backbone near the NP
n_close = 30
plot_path(backbone[-1*n_close:]+backbone[0:n_close+1], coordinates, prime_vals=prime_flags)
# closeup of the path near the NP
plot_path(path_df[-10*n_close:-1]+path_df[0:10*n_close], coordinates, prime_vals=prime_flags)
# Show the remaining cities (if any) and ALL the backbone cities
if len(remaining) > 0:
    plt.figure(figsize=(15, 10))
    plt.scatter(df_cities.loc[remaining, "X"], df_cities.loc[remaining, "Y"], s=1, c="green", alpha=0.5)
    # Backbone cities:
    plt.scatter(df_cities.loc[backbone, "X"], df_cities.loc[backbone, "Y"], s=1, c="red", alpha=0.2)
    # North pole
    plt.scatter(df_cities.iloc[0: 1, 1], df_cities.iloc[0: 1, 2], s=30, c="black")
    plt.grid(False)
    # remind what the backbone file is
    print("Backbone (red) cities are from file: ",bb_name)
    print("Remaining (green) cities will be saved and put in TSP order by themselves.")
    plt.show()

if len(remaining) > 0:
    print("There are", len(remaining), "cities remaining to add to the path;")
    n_add_bb = len(remaining)//10
    n_at_end = len(remaining) % 10
    print("add them with", n_add_bb, "additional cities in the backbone.")
else:
    print("  ***  All done - the path is complete.  ***  ")
if len(remaining) > 0:
    
    # All remaining cities:
    remaining_cities = pd.DataFrame({"CityId" : remaining})
    # Write them out to a file...
    remaining_cities.to_csv("remaining_cities_v8x.csv", index=None)

    # OK, will put these remaining cities in their own TSP path...

# Output the path to a file that we can submit (This does include the final NP !)
if len(remaining) == 0:
    submission = pd.DataFrame({"Path": path_df})
    submission.to_csv("submission.csv", index=None)
    print("Complete path, submission file written.")
else:
    print("  ***  Not a complete path, so no submission.  ***  ")
dfbb.head(5)
dfbb.iloc[1:].plot.scatter("BBtoBBlen","Length",s=3, alpha=0.25, c="blue",
                logx=True, logy=True, figsize=(10,7))
plt.title("Segment's actual Path-Length vs Backbone spacing")
plt.plot([3.0, 300.0],[3.0, 300.0], c="darkgreen")
plt.plot([3.0, 300.0],[9.0, 900.0], c="yellow")
plt.plot([3.0, 300.0],[30.0, 3000.0], c="red")
plt.show()
dfbb.iloc[1:].plot.scatter("BBtoBBlen","LenAlone",s=3, alpha=0.25, c="blue",
                logx=True, logy=True, figsize=(10,7))
plt.title("Segment's Alone-Length vs Backbone spacing")
plt.plot([3.0, 300.0],[3.0, 300.0], c="darkgreen")
plt.plot([3.0, 300.0],[9.0, 900.0], c="yellow")
plt.plot([3.0, 300.0],[30.0, 3000.0], c="red")
plt.show()
dfbb.iloc[1:].plot.scatter("BBtoBBlen","MaxRadius",s=5, alpha=0.25, c="blue",
                logx=True, logy=True, figsize=(10,7))
plt.title("Segment's actual MaxRadius vs Backbone spacing")
plt.plot([3.0, 300.0],[3.0, 300.0], c="darkgreen")
plt.plot([3.0, 300.0],[9.0, 900.0], c="yellow")
plt.plot([3.0, 300.0],[30.0, 3000.0], c="red")
plt.show()
dfbb.iloc[1:].plot.scatter("BBtoBBlen","RadAlone",s=5, alpha=0.25, c="blue",
                logx=True, logy=True, figsize=(10,7))
plt.title("Segment's Alone-Radius vs Backbone spacing")
plt.plot([3.0, 300.0],[3.0, 300.0], c="darkgreen")
plt.plot([3.0, 300.0],[9.0, 900.0], c="yellow")
plt.plot([3.0, 300.0],[30.0, 3000.0], c="red")
plt.show()
# Compare 
pd.plotting.scatter_matrix(dfbb[['BBtoBBlen','LenAlone','Length','RadAlone','MaxRadius']], figsize=(14,14),
                           diagonal='hist', range_padding=0.05, hist_kwds={'bins':100},
                           grid=True, alpha=0.5)
plt.show()
# This cell is run after the file:
#     remaining_v8C_t2500rand
# has been generated by "Concorde - Primes only" from the file created above.
if True:

    # Instead of doing all the processing above, can read in the result of the above processing,
    # will not use this when doing the Kaggle "commit"
    if False:
        dfbb = pd.read_csv("../input/santa-18-primes-tsp60/dfbb_primesTSP5000_v7B_filled.csv",
                           sep=";", index_col=0)
        # the backbone cites in this df:
        backbone = list(dfbb['CityId'])
        # Assemble the path from the dfbb
        # Start with zeroth bb (NP usually)
        path_df = [backbone[0]]
        # add the rest
        for ibb in range(1,len(backbone)):
            path_df += cities_from_str(dfbb.loc[ibb,'CityListStr'])
            path_df.append(backbone[ibb])

        
    # Get the remaining cities in TSP order, NP + 19740 others
    dftsp = pd.read_csv("../input/santa-18-primes-tsp60/remaining_v8C_t3000rand.csv")
    # Generate this backbone list of CityIds
    # Starts at the NP and ends at final city in path (close to NP).
    path_remain = list(dftsp.loc[list(dftsp['Path']),'CityId'])

    
    # Combine the two paths: Remaining path and then the Primes path.
    # The NP starts the remaining path, so skip the NP that starts the primes path
    full_path = path_remain + path_df[1:]
 
    # and zero-out remaining
    remaining = []

    print("Number of cities in the path:", len(full_path), ";    Number left to add:", len(remaining))
    
    # and output it
    submission = pd.DataFrame({"Path": full_path})
    submission.to_csv("submission.csv", index=None)
    print("Complete path, submission file written.")

    # Show the Rudolph Score results  *** Got All the Carrots ! ***
    score, carrots, length = usual_score(full_path, coordinates, prime_flags)
    # and without going back to the NP
    score_noNP, dummy1, dummy2 = usual_score(full_path[:-1], coordinates, prime_flags)

    print("Rudolph Score:", int(score), "   Carrots:", carrots, "   Length:", int(length), ".\n" +
          " Penalty frac:", int(10000*(score-length)/length)/100,
          "%   Final step to NP has distance ", int(score - score_noNP))
    
    # Show the whole combined path
    plot_path(full_path, coordinates)
# This is the final proposed path.
