import featuretools as ft

es = ft.demo.load_mock_customer(return_entityset=True)

es


feature_matrix, feature_defs = ft.dfs(entityset=es,
    target_entity="customers",
    agg_primitives=["count"],
    trans_primitives=["month"],
    max_depth=1)

feature_matrix



feature_matrix, feature_defs = ft.dfs(entityset=es,
    target_entity="customers",
    agg_primitives=["mean", "sum", "mode"],
    trans_primitives=["month", "hour"],
    max_depth=2)

feature_matrix
feature_matrix[['MEAN(sessions.SUM(transactions.amount))']]
feature_matrix[['MODE(sessions.HOUR(session_start))']]
feature_matrix, feature_defs = ft.dfs(entityset=es,
    target_entity="sessions",
    agg_primitives=["mean", "sum", "mode"],
    trans_primitives=["month", "hour"],
    max_depth=2)

feature_matrix.head(5)



feature_matrix[['customers.MEAN(transactions.amount)']].head(5)