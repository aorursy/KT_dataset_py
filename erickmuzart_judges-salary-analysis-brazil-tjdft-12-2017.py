# Importando a biblioteca pandas
import pandas as pd

# Importando o arquivo xls
# somente a planilha Contracheque -> sheet_name="Contracheque"
# descartando o cabecalho -> header=None
# descartando as 20 primeiras linhas -> skiprows=20
# e importando apenas as colunas selecionados -> usecols=[1, 2, 3, 4, 8, 14]
df = pd.read_excel("../input/Salarios_Juizes_TJDFT_122017.xls", sheet_name="Contracheque", header=None,  skiprows=20, usecols=[1, 2, 3, 4, 8, 14])

# Incluindo o cabecalho para as 5 colunas importadas
df.columns = ["Nome", "Cargo", "Lotacao", "Subsidio", "Rendimento_Bruto", "Rendimento_Liquido"]

# Exibindo os 5 primeiros registros importados
df.head()
# O pandas consegue mostrar um resumo quantitativo dos dados contidos no dataframe
# Isso vale apenas para as colunas que contenham valores numéricos
df.describe()
# Quantos magistrados estão na folha de pagamento do TJDFT?
df.count()
# É possível contar apenas uma das colunas, fazendo referência direta a
# essa coluna no dataframe e depois executando o método count()
df["Nome"].count() # ou df.Nome.count()

# E também é possível formatar a saída para um texto mais legível
print(f"O TJDFT possui {df.Nome.count()} magistrados")
# Qual o somatório do salário líquido pago a todos esses magistrados?
df.Rendimento_Liquido.sum()
# Qual a média salarial dos magistrados do TJDFT?
# O valor foi arrendodado para 2 casas decimais
print(f"Em média, um magistrado do TJDFT recebe R$ {df.Rendimento_Liquido.mean():.2f} de salário líquido por mês.")
# Quais os 5 maiores salários líquidos?
df.nlargest(5, "Rendimento_Liquido")
# E os 5 menores salários líquidos?
df.nsmallest(5, "Rendimento_Liquido")
# Vamos separar os magistrados aposentados constantes na folha de pagamento do TJDFT
# Nesse caso específico, quando o magistrado está aposentado esse termo aparece no campo Lotacao
# Isso pode não ser verdade para os dados de outros tribunais
# Vamos colocar o resultado da nossa seleção em outro dataframe
df_aposentados = df[df.Lotacao.str.contains("APOSENTADOS")]

df_aposentados.head()
# Qual o número de aposentados na folha de pagamento do TJDFT?
df_aposentados.Nome.count()
# Qual o salário médio dos aposentados do TJDFT?
print(f"No TJDFT um magistrado aposentado recebe, em média, R$ {df_aposentados.Rendimento_Liquido.mean():.2f}")
# Quais os maiores salários de magistrados aposentados?
df_aposentados.nlargest(5, "Rendimento_Liquido")
# Agora vamos separar os desembargadores constantes na folha de pagamento do TJDFT
# Nesse caso específico, quando o magistrado é desembargador esse termo aparece no campo Cargo
# Isso pode não ser verdade para os dados de outros tribunais
# Vamos colocar o resultado da nossa seleção em outro dataframe
# mas agora devemos ter cuidado para separar apenas os desembargadores que estão na ativa
df_desembargadores = df[(df.Cargo.str.contains("DESEMBARGADOR")) & ~(df.Lotacao.str.contains("APOSENTADOS"))]

df_desembargadores.head()
# Qual o número de desembargadores na folha de pagamento do TJDFT?
df_desembargadores.Nome.count()
# Qual o salário médio dos desembargadores do TJDFT?
print(f"No TJDFT um desembargador recebe, em média, R${df_desembargadores.Rendimento_Liquido.mean():.2f}")
# Quais os maiores salários de desembargadores?
df_desembargadores.nlargest(5, "Rendimento_Liquido")
# Importando a biblioteca
import matplotlib.pyplot as plt

# e determinando que os graficos serão desenhados no próprio Jupyter Notebook
%matplotlib inline

# Vamos ver a distribuição dos rendimento líquidos
df.Rendimento_Liquido.hist(bins=50)
plt.title("Distribuicao por Salarios")
plt.ylabel('Qtde de Magistrados')
plt.xlabel('Rendimento Líquido');