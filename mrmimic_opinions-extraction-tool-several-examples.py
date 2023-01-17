# Install custom library from Github

!pip install -q --no-warn-conflicts git+https://github.com/MrMimic/covid-19-kaggle



from c19 import parameters, database_utilities, text_preprocessing, embedding, query_matching, clusterise_sentences, plot_clusters, display_output



# Ugly dependencies warnings

import warnings

warnings.filterwarnings("ignore")
import os



params = parameters.Parameters(

    first_launch=True,

    database=parameters.Database(

        local_path="local_database.sqlite",

        kaggle_data_path=os.path.join(os.sep, "kaggle", "input", "CORD-19-research-challenge"),

        only_newest=True,

        only_covid=True

    ),

    preprocessing=parameters.PreProcessing(

        max_body_sentences=0,

        stem_words=False

    ),

    query=parameters.Query(

        cosine_similarity_threshold=0.8,

        minimum_sentences_kept=500,

        number_of_clusters="auto",

        k_min=3,

        k_max=10,

        min_feature_per_cluster=100

    )

)
database_utilities.create_db_and_load_articles(

    db_path=params.database.local_path,

    kaggle_data_path=params.database.kaggle_data_path,

    first_launch=params.first_launch,

    only_newest=params.database.only_newest,

    only_covid=params.database.only_covid,

    enable_data_cleaner=params.database.enable_data_cleaner)
embedding_model = embedding.Embedding(

    parquet_embedding_path=params.embedding.local_path,

    embeddings_dimension=params.embedding.dimension,

    sentence_embedding_method=params.embedding.word_aggregation_method,

    weight_vectors=params.embedding.weight_with_tfidf)
text_preprocessing.pre_process_and_vectorize_texts(

    embedding_model=embedding_model,

    db_path=params.database.local_path,

    first_launch=params.first_launch,

    stem_words=params.preprocessing.stem_words,

    remove_num=params.preprocessing.remove_numeric,

    batch_size=params.preprocessing.batch_size,

    max_body_sentences=params.preprocessing.max_body_sentences)
full_sentences_db = query_matching.get_sentences_data(

    db_path=params.database.local_path)
query = "What do we know about hydroxychloroquine to treat covid-19 disease?"
closest_sentences_df = query_matching.get_k_closest_sentences(

    query=query,

    all_sentences=full_sentences_db,

    embedding_model=embedding_model,

    minimal_number_of_sentences=params.query.minimum_sentences_kept,

    similarity_threshold=params.query.cosine_similarity_threshold)
closest_sentences_df = database_utilities.get_df_pagerank_by_doi(

    db_path=params.database.local_path, df=closest_sentences_df)
closest_sentences_df = clusterise_sentences.perform_kmean(

    k_closest_sentences_df=closest_sentences_df,

    number_of_clusters=params.query.number_of_clusters,

    k_min=params.query.k_min,

    k_max=params.query.k_max,

    min_feature_per_cluster=params.query.min_feature_per_cluster

)
plot_clusters.scatter_plot(

    closest_sentences_df=closest_sentences_df,

    query=query)
display_output.create_html_report(

    query=query,

    closest_sentences_df=closest_sentences_df,

    top_x=2,

    db_path=params.database.local_path)
query = "How neonates and pregnant women are susceptible of developing covid-19?"



closest_sentences_df = query_matching.get_k_closest_sentences(

    query=query,

    all_sentences=full_sentences_db,

    embedding_model=embedding_model,

    minimal_number_of_sentences=params.query.minimum_sentences_kept,

    similarity_threshold=params.query.cosine_similarity_threshold)



closest_sentences_df = database_utilities.get_df_pagerank_by_doi(

    db_path=params.database.local_path, df=closest_sentences_df)



closest_sentences_df = clusterise_sentences.perform_kmean(

    k_closest_sentences_df=closest_sentences_df,

    number_of_clusters=params.query.number_of_clusters,

    k_min=params.query.k_min,

    k_max=params.query.k_max,

    min_feature_per_cluster=params.query.min_feature_per_cluster

)



plot_clusters.scatter_plot(

    closest_sentences_df=closest_sentences_df,

    query=query)
display_output.create_html_report(

    query=query,

    closest_sentences_df=closest_sentences_df,

    top_x=2,

    db_path=params.database.local_path)
query = "Are smoking or pre-existing pulmonary disease (lung) risk factors for developing covid-19?"



closest_sentences_df = query_matching.get_k_closest_sentences(

    query=query,

    all_sentences=full_sentences_db,

    embedding_model=embedding_model,

    minimal_number_of_sentences=params.query.minimum_sentences_kept,

    similarity_threshold=params.query.cosine_similarity_threshold)



closest_sentences_df = database_utilities.get_df_pagerank_by_doi(

    db_path=params.database.local_path, df=closest_sentences_df)



closest_sentences_df = clusterise_sentences.perform_kmean(

    k_closest_sentences_df=closest_sentences_df,

    number_of_clusters=params.query.number_of_clusters,

    k_min=params.query.k_min,

    k_max=params.query.k_max,

    min_feature_per_cluster=params.query.min_feature_per_cluster

)



plot_clusters.scatter_plot(

    closest_sentences_df=closest_sentences_df,

    query=query)
display_output.create_html_report(

    query=query,

    closest_sentences_df=closest_sentences_df,

    top_x=2,

    db_path=params.database.local_path)
query = "Which is the cell entry receptor for SARS-cov-2?"



params.query.number_of_clusters = 1



closest_sentences_df = query_matching.get_k_closest_sentences(

    query=query,

    all_sentences=full_sentences_db,

    embedding_model=embedding_model,

    minimal_number_of_sentences=params.query.minimum_sentences_kept,

    similarity_threshold=params.query.cosine_similarity_threshold)



closest_sentences_df = database_utilities.get_df_pagerank_by_doi(

    db_path=params.database.local_path, df=closest_sentences_df)



closest_sentences_df = clusterise_sentences.perform_kmean(

    k_closest_sentences_df=closest_sentences_df,

    number_of_clusters=params.query.number_of_clusters,

    k_min=params.query.k_min,

    k_max=params.query.k_max,

    min_feature_per_cluster=params.query.min_feature_per_cluster

)



print(f"\nAnswer to the query: {query}")

print(closest_sentences_df.sort_values(by="distance", ascending=False).head(1)["raw_sentence"][0])