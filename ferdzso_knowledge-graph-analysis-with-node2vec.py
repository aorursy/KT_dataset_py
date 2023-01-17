import os

import numpy as np

import pandas as pd

import networkx as nx

from collections import Counter
files = os.listdir("../input")

files
data_parts = {}

for file_name in files:

    file_id = file_name.split(".")[0]

    data_parts[file_id] = pd.read_csv("../input/" + file_name)

    print(file_id)

    print(data_parts[file_id].shape)
def add_nodes(G, df, col, type_name):

    """Add entities to G from the 'col' column of the 'df' DataFrame. The new nodes are annotated with 'type_name' label."""

    nodes = list(df[~df[col].isnull()][col].unique())

    G.add_nodes_from([(n,dict(type=type_name)) for n in nodes])

    print("Nodes (%s,%s) were added" % (col, type_name))

    

def add_links(G, df, col1, col2, type_name):

    """Add links to G from the 'df' DataFrame. The new edges are annotated with 'type_name' label."""

    df_tmp = df[(~df[col1].isnull()) & (~df[col2].isnull())]

    links = list(zip(df_tmp[col1],df_tmp[col2]))

    G.add_edges_from([(src, trg, dict(type=type_name)) for src, trg in links])

    print("Edges (%s->%s,%s) were added" % (col1, col2, type_name))
G = nx.DiGraph()
add_nodes(G, data_parts["answers"], "answers_id", "answer")

add_nodes(G, data_parts["comments"], "comments_id", "comment")

add_nodes(G, data_parts["groups"], "groups_id", "group")

add_nodes(G, data_parts["groups"], "groups_group_type", "group_type")

add_nodes(G, data_parts["professionals"], "professionals_id", "professional")

add_nodes(G, data_parts["professionals"], "professionals_industry", "industry")

add_nodes(G, data_parts["questions"], "questions_id", "question")

add_nodes(G, data_parts["school_memberships"], "school_memberships_school_id", "school")

add_nodes(G, data_parts["students"], "students_id", "student")

add_nodes(G, data_parts["tags"], "tags_tag_id", "tag")
add_links(G, data_parts["answers"], "answers_id", "answers_question_id", "question")

add_links(G, data_parts["answers"], "answers_id", "answers_author_id", "author")

add_links(G, data_parts["comments"], "comments_id", "comments_parent_content_id", "parent_content")

add_links(G, data_parts["comments"], "comments_id", "comments_author_id", "author")

add_links(G, data_parts["group_memberships"], "group_memberships_user_id", "group_memberships_group_id", "member")

add_links(G, data_parts["groups"], "groups_id", "groups_group_type", "type")

add_links(G, data_parts["professionals"], "professionals_id", "professionals_industry", "type")

add_links(G, data_parts["questions"], "questions_id", "questions_author_id", "author")

add_links(G, data_parts["school_memberships"], "school_memberships_user_id", "school_memberships_school_id", "member")

add_links(G, data_parts["tag_questions"], "tag_questions_question_id", "tag_questions_tag_id", "tag")

add_links(G, data_parts["tag_users"], "tag_users_user_id", "tag_users_tag_id", "follow")
students = data_parts["students"]

profs = data_parts["professionals"]

students = students[~students["students_location"].isnull()]

profs = profs[~profs["professionals_location"].isnull()]
locs1 = list(students["students_location"])

locs2 = list(profs["professionals_location"])

locs = [loc.lower() for loc in locs1+locs2]

locs_unique = list(set(locs))
cnt = Counter(locs)

cnt.most_common()[:10]
new_edges = []

new_nodes = []

for loc in locs_unique:

    loc_hierarchy = loc.split(", ")

    loc_nodes = [] # due to city name duplicates in the world

    k = len(loc_hierarchy)

    for i in range(k):

        loc_nodes.append('_'.join(loc_hierarchy[i:]))

    new_nodes += loc_nodes

    loc_links = [(loc_nodes[i],loc_nodes[i+1], dict(type="location"))  for i in range(k-1)]

    new_edges += loc_links

new_nodes = list(set(new_nodes))

new_nodes = [(n, dict(type="location")) for n in new_nodes]
G.add_nodes_from(new_nodes)

G.add_edges_from(new_edges)

print(len(new_edges), len(new_nodes))
list(G.in_edges("united kingdom"))[:5]
list(G.in_edges("england_united kingdom"))[:5]
students["students_location"] = students["students_location"].apply(lambda x: "_".join(x.lower().split(", ")))

profs["professionals_location"] = profs["professionals_location"].apply(lambda x: "_".join(x.lower().split(", ")))
add_links(G, students, "students_id", "students_location", "location")

add_links(G, profs, "professionals_id", "professionals_location", "location")
def encode_graph(G):

    """Encode the nodes of the network into integers"""

    nodes = [(n,d.get("type",None)) for n, d in G.nodes(data=True)]

    nodes_df = pd.DataFrame(nodes, columns=["id","type"]).reset_index()

    node2idx = dict(zip(nodes_df["id"],nodes_df["index"]))

    edges = [(node2idx[src], node2idx[trg], d.get("type",None)) for src, trg, d in G.edges(data=True)]

    edges_df = pd.DataFrame(edges, columns=["src","trg","type"])

    return nodes_df, edges_df
print(G.number_of_nodes(), G.number_of_edges())

G.remove_nodes_from(list(nx.isolates(G)))

print(G.number_of_nodes(), G.number_of_edges())
nodes_df, edges_df = encode_graph(G)

len(nodes_df), len(edges_df)
print(nodes_df.head())

print(nodes_df["type"].value_counts())

nodes_df.to_csv("knowledge_graph_nodes.csv", index=False)
print(edges_df.head())

print(edges_df["type"].value_counts())

edges_df[["src","trg"]].to_csv("knowledge_graph_edges.csv", index=False, header=False, sep=" ")
edge_list = list(zip(edges_df["src"],edges_df["trg"]))

edge_list[:5]
KG = nx.Graph(edge_list)

KG.number_of_nodes(), KG.number_of_edges()
largest_cc = max(nx.connected_components(KG), key=len)

KG = nx.subgraph(KG, largest_cc)

KG.number_of_nodes(), KG.number_of_edges()
%%time

from node2vec import Node2Vec

n2v_obj = Node2Vec(KG, dimensions=10, walk_length=5, num_walks=10, p=1, q=1, workers=1)
%%time

n2v_model = n2v_obj.fit(window=3, min_count=1, batch_words=4)
%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE



def get_embeddings(model, nodes):

    """Extract representations from the node2vec model"""

    embeddings = [list(model.wv.get_vector(n)) for n in nodes]

    embeddings = np.array(embeddings)

    print(embeddings.shape)

    return embeddings



def dim_reduction(embeddings, labels, frac=None, tsne_obj=TSNE(n_components=2)):

    """Dimensionality reduction with t-SNE. Sampling random instances is supported."""

    N = len(embeddings)

    print(N)

    if frac != None:

        idx = np.random.randint(N, size=int(N*frac))

        X = embeddings[idx,:]

        X_labels = [labels[i] for i in idx]

    else:

        X = embeddings

        X_labels = labels

    X_embedded = tsne_obj.fit_transform(X)

    print("t-SNE object was trained on %i records!" % X.shape[0])

    print(X_embedded.shape)

    return X_embedded, X_labels



def visu_embeddings(X_embedded, X_labels=None, colors = ['r','b']):

    if X_labels != None:

        label_map = {}

        for i, l in enumerate(usr_tsne_lab):

            if not l in label_map:

                label_map[l] = []

            label_map[l].append(i)

        fig, ax = plt.subplots(figsize=(15,15))

        for i, lab in enumerate(label_map.keys()):

            print(lab)

            idx = label_map[lab]

            x = list(X_embedded[idx,0])

            y = list(X_embedded[idx,1])

            #print(len(x),len(y))

            ax.scatter(x, y, c=colors[i], label=lab, alpha=0.5, edgecolors='none')

        plt.legend()

    else:

        plt.figure(figsize=(15,15))

        x = list(X_embedded[:,0])

        y = list(X_embedded[:,1])

        plt.scatter(x, y, alpha=0.5)
stud_users = list(nodes_df[nodes_df["type"] == "student"]["index"])

prof_users = list(nodes_df[nodes_df["type"] == "professional"]["index"])

print(len(stud_users), len(prof_users))

stud_users = list(set(stud_users).intersection(set(KG.nodes())))

prof_users = list(set(prof_users).intersection(set(KG.nodes())))

print(len(stud_users), len(prof_users))

stud_users = [str(item) for item in stud_users]

prof_users = [str(item) for item in prof_users]
users = stud_users + prof_users

usr_emb = get_embeddings(n2v_model, users)

usr_labs = ['student'] * len(stud_users) +  ['professional'] * len(prof_users)
%%time

usr_tsne_emb, usr_tsne_lab = dim_reduction(usr_emb, usr_labs, frac=0.5)
visu_embeddings(usr_tsne_emb, usr_tsne_lab)