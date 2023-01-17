
from __future__ import print_function
from __future__ import division
import math


def jaro_similarity(s1, s2):
    """
    Computes the Jaro similarity between 2 sequences from:

        Matthew A. Jaro (1989). Advances in record linkage methodology
        as applied to the 1985 census of Tampa Florida. Journal of the
        American Statistical Association. 84 (406): 414–20.

    The Jaro distance between is the min no. of single-character transpositions
    required to change one word into another. The Jaro similarity formula from
    https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance :

        jaro_sim = 0 if m = 0 else 1/3 * (m/|s_1| + m/s_2 + (m-t)/m)

    where:
        - |s_i| is the length of string s_i
        - m is the no. of matching characters
        - t is the half no. of possible transpositions.
    """
    # First, store the length of the strings
    # because they will be re-used several times.
    len_s1, len_s2 = len(s1), len(s2)

    # The upper bound of the distanc for being a matched character.
    match_bound = math.floor( max(len(s1), len(s2)) / 2 ) - 1

    # Initialize the counts for matches and transpositions.
    matches = 0  # no.of matched characters in s1 and s2
    transpositions = 0  # no. transpositions between s1 and s2

    # Iterate through sequences, check for matches and compute transpositions.
    for ch1 in s1:     # Iterate through each character.
        if ch1 in s2:  # Check whether the
            pos1 = s1.index(ch1)
            pos2 = s2.index(ch1)
            if(abs(pos1-pos2) <= match_bound):
                matches += 1
                if(pos1 != pos2):
                    transpositions += 1

    if matches == 0:
        return 0
    else:
        return 1/3 * ( matches/len_s1 +
                       matches/len_s2 +
                      (matches-transpositions//2) / matches
                     )


def jaro_winkler_similarity(s1, s2, p=0.1, max_l=None):
    """
    The Jaro Winkler distance is an extension of the Jaro similarity in:

        William E. Winkler. 1990. String Comparator Metrics and Enhanced
        Decision Rules in the Fellegi-Sunter Model of Record Linkage.
        Proceedings of the Section on Survey Research Methods.
        American Statistical Association: 354–359.

    such that:

        jaro_winkler_sim = jaro_sim + ( l * p * (1 - jaro_sim) )

    where,

        - jaro_sim is the output from the Jaro Similarity, see jaro_similarity()
        - l is the length of common prefix at the start of the string
            - this implementation provides an upperbound for the l value
              to keep the prefixes
        - p is the constant scaling factor to overweigh common prefixes.
          The Jaro-Winkler similarity will fall within the [0, 1] bound,
          given that max(p)<=0.25 , default is p=0.1 in Winkler (1990)

    """
    # Compute the Jaro similarity
    jaro_sim = jaro_similarity(s1, s2)

    # Initialize the upper bound for the no. of prefixes.
    # if user did not pre-define the upperbound, use length of s1
    max_l = max_l if max_l else len(s1)

    # Compute the prefix matches.
    l = 0
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            l += 1
        else:
            break
        if l == max_l:
            break
    # Return the similarity value as described in docstring.
    return jaro_sim + ( l * p * (1 - jaro_sim) )

string_distance_examples = [
    ("rain", "shine"), ("abcdef", "acbdef"), ("language", "lnaguaeg"),
    ("language", "lnaugage"), ("language", "lngauage")]

for s1, s2 in string_distance_examples:
    print("Jaro similarity btwn '%s' and '%s':" % (s1, s2),
          jaro_similarity(s1, s2))
def rosetta_jaro_similarity(s1, s2):
    
     # First, store the length of the strings
    # because they will be re-used several times.
    len_s1, len_s2 = len(s1), len(s2)

    # The upper bound of the distanc for being a matched character.
    match_bound = math.floor( max(len(s1), len(s2)) / 2 ) - 1

    # Initialize the counts for matches and transpositions.
    matches = 0  # no.of matched characters in s1 and s2
    transpositions = 0  # no. transpositions between s1 and s2
    
    # Initialize a binary array to track whether
    # characters between the sequence matches
    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2
    
    # Populate the no. of matches first.
    for i in range(len_s1):
        # Give the bound of where to search for the possible transpositions
        start = max(0, i-match_bound)
        end = min(i+match_bound+1, len_s2)
        # Iterate through the s2[start:end] positions of 
        # the second string and check whether it matches
        # with the s1[i]
        for j in range(start, end):
            # If it's already matching or
            # if the characters don't match
            if s1_matches[i]:
                continue
            if (s1[i] != s2[j]):
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
            
    # Compute the transpositions.
    j = 0 # Use j to track the position of s2
    for i in range(len_s1): # Use i to track position of s1
        if not s1_matches[i]:
            continue
        if j >= len_s2:
            break
        while not s2_matches[j]:
            j += 1
            if j >= len_s2:
                break
        if j >= len_s2:
            break
        if s1[i] != s2[j]:
            transpositions += 1
        j += 1
        if j >= len_s2:
            break

    
    if matches == 0:
        return 0
    else:
        return 1/3 * ( matches/len_s1 +
                       matches/len_s2 +
                      (matches-transpositions//2) / matches
                     )
string_distance_examples = [
    ("rain", "shine"), ("abcdef", "acbdef"), ("language", "lnaguaeg"),
    ("language", "lnaugage"), ("language", "lngauage")]

for s1, s2 in string_distance_examples:
    print("Jaro similarity btwn '%s' and '%s':" % (s1, s2),
          jaro_similarity(s1, s2))
    print("Rosetta Jaro similarity btwn '%s' and '%s':" % (s1, s2),
          rosetta_jaro_similarity(s1, s2))
    print()
_table5 = """billy billy	1.000	1.000	1.000
billy bill	0.967	0.933	0.800
billy blily	0.947	0.933	0.600
massie massey	0.944	0.889	0.600
yvette yevett	0.911	0.889	0.600
billy bolly	0.893	0.867	0.600
dwayne duane	0.858	0.822	0.400
dixon dickson	0.853	0.791	0.200
billy susan	0.000	0.000	0.000"""

table5 = []
for row in _table5.split('\n'):
    s1s2, w, j, _ = row.split('\t')
    j = float(j)
    s1, s2 = s1s2.split()
    table5.append((s1,s2,j))
    
    
print('nltk\trosetta\tpaper')
for s1, s2, j in table5:
    row = [jaro_similarity(s1, s2), rosetta_jaro_similarity(s1, s2), j]
    print('\t'.join(["%0.3f" % s for s in row]))


