import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS Transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month 
            ORDER BY year, month
        """

# Configuramos el parametro max_gb_scanned para poder leer la totalidad de la base de datos
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=24)
# Importamos la librería matplotlib para poder gráficar la query realizada
import matplotlib.pyplot as plt

#Graficamos los datos para poder compararla con la gráfica aportada.
with plt.style.context('classic'):
    plt.plot(transactions_per_month.Transactions)
    plt.title("Monthly Bitcoin Transactions")
    plt.grid(True)
    plt.xlabel('Month')
    plt.ylabel('NºTransactions')
    plt.legend(loc=0)