import matplotlib.pyplot as plt

import networkx as nx

import numpy as np

import scipy.sparse as sp



def laplacian_embedding(g_nx, dim):

    """Compute the smallest-but-1 eigenvectors of the laplacian normalized by the variance."""

    L = nx.laplacian_matrix(g_nx).astype("float64")

    # returns ordered by the smallest eigenvalues

    w, v = sp.linalg.eigsh(L, k=dim + 1)

    return np.divide(v[:, 1:], np.sqrt(w[1:]))[::-1]
!ls /kaggle/input/cord-19-citation-network
import pandas as pd



citations = pd.read_parquet("/kaggle/input/cord-19-citation-network/citations.parquet")

citations
# generate the master mapping of names, this is slow in pandas and is probably better suited for spark

title_index = (

    pd.concat(

        [

            citations.rename(columns={"src_citation_id": "citation_id", "paper": "title"}),

            citations.rename(columns={"dst_citation_id": "citation_id", "citation": "title"})

        ], 

        ignore_index=True,

        sort=False

    )[["citation_id", "title"]]

    .groupby("citation_id")["title"]

    .first()

)



title_index
G = nx.from_pandas_edgelist(

    citations, 

    source="src_citation_id",

    target="dst_citation_id",

    create_using=nx.DiGraph

)

print(nx.info(G))
%time assert all(x == y for x, y in zip(G.nodes, G.to_undirected().nodes))
%time pagerank = pd.DataFrame(nx.pagerank(G).items(), columns=["citation_id", "pagerank"]).sort_values(by="pagerank", ascending=False)

pagerank.to_csv("pagerank.csv")
pd.set_option('display.max_colwidth', 120)



def top_titles(df, index, rank):

    # index[citation_id, title]

    return (

        pd.DataFrame(index)

        .join(df.set_index("citation_id"))

        .sort_values(by=rank, ascending=False)

    ).reset_index()
ranked_citations = top_titles(pagerank, title_index, "pagerank")

ranked_citations[["title", "pagerank"]][:10]
pagerank_indirect = (

    pd.DataFrame(nx.pagerank(G.to_undirected()).items(), columns=["citation_id", "pagerank"])

    .sort_values(by="pagerank", ascending=False)

)

top_titles(pagerank_indirect, title_index, "pagerank")[["title", "pagerank"]][:10]
# convert the graph to a undirected graph naively

# See this link for an interesting way to preserve direction:

# https://cs.stackexchange.com/a/43010

%time emb = laplacian_embedding(nx.Graph(G), dim=2)

plt.scatter(emb[:, 0], emb[:,1])

plt.show()
%time emb = laplacian_embedding(G.to_undirected(), dim=8)

print(f"{emb.shape} -> {emb.shape[0]*emb.shape[1]*8/10**6} mb")

# TODO: T-SNE
def embedding_matches(G, emb, title_index, pattern, k=None):

    emb_df = (

        pd.DataFrame(G.nodes(), columns=["citation_id"])

        .join(pd.DataFrame(emb))

        .set_index("citation_id")

    )

    indices = title_index.str.lower().str.match(pattern)

    return emb_df.loc[indices].join(title_index)
env_df = embedding_matches(G, emb, title_index,"enviro.*")

env_df
from sklearn.neighbors import NearestNeighbors





def measure_graph(ann, data, n_neighbors):

    # all nearest neighbors

    index = ann.fit(data)

    smat = index.kneighbors_graph(data, n_neighbors)

    g = nx.from_scipy_sparse_matrix(smat)

    print(nx.info(g))

    nx.draw_spring(g, node_size=10)

    return g



ann = NearestNeighbors(algorithm="ball_tree", n_jobs=-1)

env_emb = env_df.iloc[:,:8].values
g10 = measure_graph(ann, env_emb, 10)
measure_graph(ann, env_emb, 15)
g20 = measure_graph(ann, env_emb, 20)
measure_graph(ann, env_emb, 25)
g30 = measure_graph(ann, env_emb, 30)
measure_graph(ann, env_emb, 50)
g10_small = g10.subgraph(min(nx.connected_components(g10), key=len))

g10_large = g10.subgraph(max(nx.connected_components(g10), key=len))
nx.draw_spectral(g10_small)
nx.draw_spectral(g10_large)
nx.draw_spectral(g20)
nx.draw_spectral(g30)
def show_ranked(df, g, nx_func, output=None, k=None, **kwargs):

    score = nx_func(g)

    

    if not kwargs.get("node_size"):

        kwargs["node_size"] = 30

    nx.draw(g, node_color=list(map(score.get, g.nodes())), **kwargs)

    

    indices = sorted(score, key=score.get, reverse=True)

    score_df = pd.DataFrame(map(score.get, indices), columns=["score"])

    res_df = pd.concat([df.iloc[indices].reset_index(), score_df], axis=1)



    if output:

        res_df.to_csv(output)

    return res_df[["title", "score"]][:k]
show_ranked(env_df, g10_small, nx.pagerank, node_size=300, output="pagerank_g10_small.csv")
%time show_ranked(env_df, g10_large, nx.pagerank, output="pagerank_g10_large.csv")
show_ranked(env_df, g20, nx.pagerank, output="pagerank_g20.csv")
show_ranked(env_df, g20, nx.pagerank, output="pagerank_g20.csv")
show_ranked(env_df, g30, nx.pagerank, output="pagerank_g30.csv")
show_ranked(env_df, g10_small, nx.closeness_centrality, output="closeness_g10_small.csv", node_size=300)
%time show_ranked(env_df, g10_large, nx.closeness_centrality, output="closeness_g10_large.csv")
show_ranked(env_df, g20, nx.closeness_centrality, output="closeness_g20.csv")
show_ranked(env_df, g30, nx.closeness_centrality, output="closeness_g30.csv")
show_ranked(env_df, g10_small, nx.betweenness_centrality, output="betweenness_g10_small.csv", node_size=300)
%time show_ranked(env_df, g10_large, nx.betweenness_centrality, output="betweenness_g10_large.csv")
show_ranked(env_df, g20, nx.betweenness_centrality, output="betweenness_g20.csv")
show_ranked(env_df, g30, nx.betweenness_centrality, output="betweenness_g30.csv")
emb.shape
pattern = ".*(corona|enviro|sewage|water|food).*"

virus_env_df = embedding_matches(G, emb, title_index, pattern)

virus_env_emb = virus_env_df.iloc[:,:8].values

virus_env_df
# virus10 = measure_graph(ann, virus_env_emb, 10)