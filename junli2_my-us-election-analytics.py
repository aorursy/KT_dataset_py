import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets 
# preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved 
# outside of the current session
usa_2016_presidential_election_by_county = pd.read_csv('/kaggle/input/us-elections-dataset/usa-2016-presidential-election-by-county.csv', sep=';')
print(f"Totally, there are {len(usa_2016_presidential_election_by_county)} records")

usa_2016_presidential_election_by_county.head()
for k in usa_2016_presidential_election_by_county.keys():
    print(k)
# State
# County
df = usa_2016_presidential_election_by_county.dropna(subset=[
    "Votes16 Clintonh", "Votes16 Trumpd", 
    "Republicans 2016", "Democrats 2016",
    "Republicans 2012", "Republicans 2008", 
    "Democrats 2012", "Democrats 2008", "Votes"])
n_dem = df["Votes16 Clintonh"].sum() 
n_rep = df["Votes16 Trumpd"].sum()
p_dem = n_dem / (n_dem + n_rep)
p_rep = n_rep / (n_dem + n_rep)
print(f"Votes for DEM {n_dem}, probability {p_dem:.4f}")
print(f"Votes for REP {n_rep}, probability {p_rep:.4f}")

ent = - (p_dem * np.log2(p_dem) + p_rep * np.log2(p_rep)).sum()
print(f"Entropy: {ent:.4f}")
print(f"""This means if you store all the election ballots in 2016, the MINIMUM file size
cannot be less than {ent * (n_dem+n_rep):.2f} bits""")
# check out all California 
df[df["State"] == "California"]
# Let us summarise the entropy computation and report in a function
def exam_votes(df_i):
    n_dem = df_i["Votes16 Clintonh"].sum() 
    n_rep = df_i["Votes16 Trumpd"].sum()
    p_dem = n_dem / (n_dem + n_rep)
    p_rep = n_rep / (n_dem + n_rep)
    print(f"2016 Vote Statistics {n_dem + n_rep} votes in {len(df_i)} counties")
    print(f"Votes for DEM {n_dem}, probability {p_dem:.4f}")
    print(f"Votes for REP {n_rep}, probability {p_rep:.4f}")
    ent = - (p_dem * np.log2(p_dem) + p_rep * np.log2(p_rep)).sum()
    print(f"Entropy: {ent:.4f}")
    print(f"""This means if you store all the election ballots in 2016, the MINIMUM file size
cannot be less than {ent * (n_dem+n_rep):.2f} bits""")
    return ent, p_dem, p_rep, n_dem, n_rep

ent, p_dem, p_rep, n_dem, n_rep = exam_votes(df[df["State"] == "California"])
import plotly.express as px
fig = px.scatter_geo(df, lat="lat", lon="lon", color="Republicans 2016", hover_name="County", size="Votes")#, 
# you can try to remove "size" to get quicker rendering and get smaller counties more visible
fig.show()
px.scatter(df, x="Republicans 2016", y="Democrats 2016", hover_name="County")
# Get the record of the county "District of Columbia, District of Columbia"
df[df["County"] == "District of Columbia, District of Columbia"] # only one record
# Let's to the entropy computation
_ = exam_votes(df[df["County"] == "District of Columbia, District of Columbia"])
df["Republicans Won 2016"] = df["Democrats 2016"] < df["Republicans 2016"]
df["Republicans Won 2012"] = df["Democrats 2012"] < df["Republicans 2012"]
df["Republicans Won 2008"] = df["Democrats 2008"] < df["Republicans 2008"]
# Check the 2016 results
df["Republicans Won 2016"].value_counts(normalize=True)
prob = df["Republicans Won 2016"].value_counts(normalize=True)
prob = np.array(prob)
print(f"Distribution of *repub won* w.r.t. county is [True (Rep Won), False (Dem Won)]={prob}")
ent = - (prob * np.log2(prob)).sum()
print(f"Entropy is {ent:.4f}")
# summerise the previous analysis into a county based function

def exam_counties(df, verbose=True):
    prob = df["Republicans Won 2016"].value_counts(normalize=True)
    prob = np.array(prob)
    ent = - (prob * np.log2(np.maximum(prob, 1e-6))).sum()
    if verbose:
        print(f"Distribution of *repub won* w.r.t. county is [True (Rep Won), False (Dem Won)]={prob}")
        print(f"Entropy is {ent:.4f}")
    return ent
# Let's try states ...
states = df["State"].value_counts()
states

exam_counties(df[df["State"]=="Georgia"], verbose=False)
total_ent = 0
num_counties = 0
for k, v in states.iteritems():
    ent = exam_counties(df[df["State"]==k], verbose=False) # in this particular state
    print(f"State {k} has {v} counties, result entropy {ent:.3f}")
    total_ent += v * ent
    num_counties += v
    
print(f"Weighted sum of entropies {total_ent/num_counties :.3f}")
ent = exam_counties(df[df["ST"]=="CA"]) # in this particular state the result is more unpredicable (in terms of counties)
# Examine the education information.
df[["Less Than High School Diploma", "At Least High School Diploma",
    "At Least Bachelors's Degree","Graduate Degree"]]
fig = px.scatter(df, x="At Least Bachelors's Degree", y="Democrats 2016", 
                 color="Republicans Won 2016", color_discrete_sequence=['red','blue'])
fig.show()
# fig = px.scatter(df, x="At Least Bachelors's Degree", y="Democrats 2016", color="Republicans Won 2016",
#                  color_discrete_sequence=['red','blue'], size="Votes")
# fig.show()
# You can check the size="Votes" to see how significant the individual counties are

df["More Than 30p Bachelors"] = df["At Least Bachelors's Degree"] > 30
# We do the same calculation as above
total_ent = 0
num_counties = 0
attr = "More Than 30p Bachelors"
for k, v in df[attr].value_counts().iteritems():
    ent = exam_counties(df[df[attr]==k], verbose=False) # in this particular state
    print(f"there are {v} counties where {attr} is {k}, result entropy {ent:.3f}")
    total_ent += v * ent
    num_counties += v
    
print(f"Weighted sum of entropies {total_ent/num_counties :.3f}")

# recall that the original entropy is ... (copied from above)
prob = df["Republicans Won 2016"].value_counts(normalize=True)
prob = np.array(prob)
print(f"Distribution of *repub won* w.r.t. county is [True (Rep Won), False (Dem Won)]={prob}")
ent0 = - (prob * np.log2(prob)).sum()
print(f"Entropy is {ent0:.4f}")
print(f"Info Gain: {ent0 - total_ent/num_counties:.4f}")
fig = px.scatter(df, x="White (Not Latino) Population", y="Democrats 2016", color="Republicans Won 2016",
                color_discrete_sequence=['red','blue'])
fig.show()
df["White (Not Latino) Population Is Greater Than 60p"] = df["White (Not Latino) Population"] > 60
# By now we have a pattern of measuring the information gain
# 1. get the unique values:
#    True/False, 
#    Texas, Georgia, Virginia ...
#    and get the sub-populations (of counties)
# 2. compute the entropy of the sub-populations
# 3. get the weighted sum of entropy
# 4. compare
# 

def compute_weighted_sub_entropy(df, attr, verbose=True):
    total_ent = 0
    num_counties = 0
    for k, v in df[attr].value_counts().iteritems():
        ent = exam_counties(df[df[attr]==k], verbose=False) # in this particular sub-population
        if verbose:
            print(f"there are {v} counties where {attr} is {k}, result entropy {ent:.3f}")
        total_ent += v * ent
        num_counties += v
    
    weighted_ent = total_ent/num_counties
    if verbose:
        print(f"Weighted sum of entropies {weighted_ent:.3f}")
    return weighted_ent

weighted_ent = compute_weighted_sub_entropy(df, "White (Not Latino) Population Is Greater Than 60p")
print(f"Info Gain: {ent0 - weighted_ent:.4f}")
attributes = ["White (Not Latino) Population", 
    "African American Population",
    "Native American Population",
    "Asian American Population", 
    "Latino Population",
    "Less Than High School Diploma",
    "At Least High School Diploma",
    "At Least Bachelors's Degree",
    "Graduate Degree",
    "School Enrollment",
    "Median Earnings 2010",
    "Children Under 6 Living in Poverty",
    "Adults 65 and Older Living in Poverty",
    "Preschool.Enrollment.Ratio.enrolled.ages.3.and.4",
    "Poverty.Rate.below.federal.poverty.threshold",
    "Gini.Coefficient",
    "Child.Poverty.living.in.families.below.the.poverty.line",
    "Management.professional.and.related.occupations",
    "Service.occupations",
    "Sales.and.office.occupations",
    "Farming.fishing.and.forestry.occupations",
    "Construction.extraction.maintenance.and.repair.occupations",
    "Production.transportation.and.material.moving.occupations",
    "Median Age",
    "Poor.physical.health.days",
    "Poor.mental.health.days",
    "Low.birthweight",
    "Teen.births",
    "Children.in.single.parent.households",
    "Adult.smoking",
    "Adult.obesity",
    "Diabetes",
    "Sexually.transmitted.infections",
    "HIV.prevalence.rate",
    "Uninsured",
    "Unemployment",
    "Violent.crime",
    "Homicide.rate",
    "Injury.deaths",
    "Infant.mortality"]
new_attributes = []
for a in attributes:
    new_a = "Quant4." + a
    df[new_a] = pd.qcut(df[a], q=4, labels=["q1", "q2", "q3", "q4"])
    new_attributes.append(new_a)


vote_info = [
    "Votes16 Trumpd",
    "Votes16 Clintonh",
    "State",
    "ST",
    "Fips",
    "County",
    "Precincts",
    "Votes",
    "Democrats 08 (Votes)",
    "Democrats 12 (Votes)",
    "Republicans 08 (Votes)",
    "Republicans 12 (Votes)",
    "Republicans 2016",
    "Democrats 2016",
    "Green 2016",
    "Libertarians 2016",
    "Republicans 2012",
    "Republicans 2008",
    "Democrats 2012",
    "Democrats 2008"]
df_new = df[new_attributes]
df_new.dropna('columns', 'any')
df_new
# The compute entropy and info_gain are copied from our exercise notebook,
# the procedure is as explained in the analysis steps above. 
def compute_entropy(y):
    """
    :param y: The data samples of a discrete distribution
    """
    if len(y) < 2: #  a trivial case
        return 0
    freq = np.array( y.value_counts(normalize=True) )
    return -(freq * np.log2(freq + 1e-6)).sum() # the small eps for 
    # safe numerical computation 
    
def compute_info_gain(samples, attr, target):
    values = samples[attr].value_counts(normalize=True)
    split_ent = 0
    for v, fr in values.iteritems():
        index = samples[attr]==v
        sub_ent = compute_entropy(target[index])
        split_ent += fr * sub_ent
    
    ent = compute_entropy(target)
    return ent - split_ent

class TreeNode:
    """
    A recursively defined data structure to store a tree.
    Each node can contain other nodes as its children
    """
    def __init__(self, node_name="", min_sample_num=10, default_decision=None):
        self.children = {} # Sub nodes --
        # recursive, those elements of the same type (TreeNode)
        self.decision = None # Undecided
        self.split_feat_name = None # Splitting feature
        self.name = node_name
        self.default_decision = default_decision
        self.min_sample_num = min_sample_num

    def pretty_print(self, prefix=''):
        if self.split_feat_name is not None:
            for k, v in self.children.items():
                v.pretty_print(f"{prefix}:When {self.split_feat_name} is {k}")
                #v.pretty_print(f"{prefix}:{k}:")
        else:
            print(f"{prefix}:{self.decision}")

    def predict(self, sample):
        if self.decision is not None:
            # uncomment to get log information of code execution
            print("Decision:", self.decision)
            return self.decision
        else: 
            # this node is an internal one, further queries about an attribute 
            # of the data is needed.
            attr_val = sample[self.split_feat_name]
            child = self.children[attr_val]
            # uncomment to get log information of code execution
            print("Testing ", self.split_feat_name, "->", attr_val)

            # [Exercise]
            # Insert your code here
            return child.predict(sample)

    def fit(self, X, y):
        """
        The function accepts a training dataset, from which it builds the tree 
        structure to make decisions or to make children nodes (tree branches) 
        to do further inquiries
        :param X: [n * p] n observed data samples of p attributes
        :param y: [n] target values
        """
        if self.default_decision is None:
            self.default_decision = y.mode()[0]
            
            
        print(self.name, "received", len(X), "samples")
        if len(X) < self.min_sample_num:
            # If the data is empty when this node is arrived, 
            # we just make an arbitrary decision
            if len(X) == 0:
                self.decision = self.default_decision
                print("DECESION", self.decision)
            else:
                self.decision = y.mode()[0]
                print("DECESION", self.decision)
            return
        else: 
            unique_values = y.unique()
            if len(unique_values) == 1:
                self.decision = unique_values[0]
                print("DECESION", self.decision)
                return
            else:
                info_gain_max = 0
                for a in X.keys(): # Examine each attribute
                    aig = compute_info_gain(X, a, y)
                    if aig > info_gain_max:
                        # [Exercise]
                        # Insert your code here
                        info_gain_max = aig
                        self.split_feat_name = a
                print(f"Split by {self.split_feat_name}, IG: {info_gain_max:.2f}")
                self.children = {}
                for v in X[self.split_feat_name].unique():
                    index = X[self.split_feat_name] == v
                    self.children[v] = TreeNode(
                        node_name=self.name + ":" + self.split_feat_name + "==" + str(v),
                        min_sample_num=self.min_sample_num,
                        default_decision=self.default_decision)
                    self.children[v].fit(X[index], y[index])

# Test tree building
data = df[new_attributes].dropna('columns', 'any')
target = df["Republicans Won 2016"]

t = TreeNode(min_sample_num=50)
t.fit(data, target)
corr = 0
err_fp = 0
err_fn = 0
for (i, ct), tgt in zip(data.iterrows(), target):
    a = t.predict(ct)
    if a and not tgt:
        err_fp += 1
    elif not a and tgt:
        err_fn += 1
    else:
        corr += 1
        

corr, err_fp, err_fn