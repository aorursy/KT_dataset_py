# importar el paquete helper
import bq_helper

# preparar un helper para trabajar con el data set de transacciones de bitcoin
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="bitcoin_blockchain")

# consulta a ejecutar. Básicamente agruparemos todas las transacciones por año y luego por mes y luego contaremos cuántos id de transacción hay.
# en este caso consideramos que cada transaction_id del data set es una transacción distinta.
query = """ SELECT COUNT(transaction_id) AS numeroTransacciones,
                EXTRACT(MONTH FROM TIMESTAMP_MILLIS(timestamp)) AS mes,
                EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) AS anyo
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY anyo, mes 
            ORDER BY anyo, mes
        """

# ejecutamos la consulta con el helper. El helper permite establecer un tamaño máximo (en gigas) para la consulta.
transacciones_mensuales = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=30)
# consultamos los 10 primeros registros y los 10 últimos
print(transacciones_mensuales.head(10))
print(transacciones_mensuales.tail(10))
# graficaremos los resultados obtenidos para poder comparar los datos con el análisis del cliente
import matplotlib.pyplot as plt

plt.plot(transacciones_mensuales.numeroTransacciones)
plt.title("Transacciones Mensuales de Bitcoin")