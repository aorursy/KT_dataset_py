import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
# importação da primeira planilha - para treinamento
resenhas_df = pd.read_excel('../input/satisfacao-apps-bancos/Satisfao com App.xlsx')
resenhas_df.info()
# identificar valores nulos
resenhas_df.isna().sum()
# excluir os registros com valores missing
resenhas_df.dropna(inplace=True)
# A criação de variáveis é uma etapa de feature engineering. Como a variável aqui será utilizada
# para criação de outro dataset, optou-se por já disponibilizá-la.
resenhas_df['qtde_caracteres']  = resenhas_df['Comentario'].str.len().fillna(1).astype(int)
resenhas_df['qtde_palavras']  = resenhas_df['Comentario'].fillna(' ').str.split().str.len()
resenhas_df['qtde_palavras']  = resenhas_df['qtde_palavras'].fillna(0).astype(int)
# alterar as variáveis relativas aos sentimentos atribuídos às resenhas dos usuários em maiúsculas
ls_cols_sentimentos = [col for col in resenhas_df.columns if 'Elogio' in col or 'Reclamação' in col or 'Classificável' in col]
resenhas_df[ls_cols_sentimentos] = resenhas_df[ls_cols_sentimentos].apply(lambda x: x.str.upper())

# unificar a ausência do til
for var in ls_cols_sentimentos:
    resenhas_df[var]        = resenhas_df[var].map({'NÃO': 'NAO', 'SIM': 'SIM'})
# criar um conjunto de dados para cada instituição
resenha_Bradesco = resenhas_df[resenhas_df.Instituição == 'Banco Bradesco'].drop('Instituição', axis = 1)
resenha_Itau     = resenhas_df[resenhas_df.Instituição == 'Banco Itau'].drop('Instituição', axis = 1)
resenha_B_Brasil = resenhas_df[resenhas_df.Instituição == 'Banco do Brasil'].drop('Instituição', axis = 1)
resenha_Caixa    = resenhas_df[resenhas_df.Instituição == 'Caixa Econômica Federal'].drop('Instituição', axis = 1)
resenha_Nubank   = resenhas_df[resenhas_df.Instituição == 'Nubank'].drop('Instituição', axis = 1)
# amostra dos dados
resenhas_df.head()
# definir padrão dos gráficos
sns.set_context('talk')
registros_por_instituicao = resenhas_df.groupby('Instituição').count()

plt.figure(figsize = (22, 8))
ax = sns.barplot(x = registros_por_instituicao.index,
                 y = 'Comentario',
                 data = registros_por_instituicao)
nota_media = resenhas_df.groupby('Instituição').mean()

plt.figure(figsize = (22, 8))
sns.barplot(x = registros_por_instituicao.index, y = 'Classificação', data = nota_media)
plt.show()
plt.figure(figsize = (22, 8))
ax = sns.countplot(x='Classificação', data=resenhas_df)
ax.set(title = 'Quantidade de Cada Avaliação', xlabel='Avaliação Atribuída', ylabel='Quantidade de Avaliações')
plt.show()
plt.figure(figsize = (22, 8))

ax = sns.countplot(x='Classificação', data=resenha_Bradesco)
ax.set(title = 'Quantidade de Cada Avaliação do Bradesco', xlabel='Avaliação Atribuída', ylabel='Quantidade de Avaliações')
plt.show()
plt.figure(figsize = (22, 8))

ax = sns.countplot(x='Classificação', data=resenha_Itau)
ax.set(title = 'Quantidade de Cada Avaliação do Itau', xlabel='Avaliação Atribuída', ylabel='Quantidade de Avaliações')
plt.show()
plt.figure(figsize = (22, 8))

ax = sns.countplot(x='Classificação', data=resenha_B_Brasil)
ax.set(title = 'Quantidade de Cada Avaliação do Banco do Brasil', xlabel='Avaliação Atribuída', ylabel='Quantidade de Avaliações')
plt.show()
plt.figure(figsize = (22, 8))

ax = sns.countplot(x='Classificação', data=resenha_Caixa)
ax.set(title = 'Quantidade de Cada Avaliação da Caixa Econômica Federal', xlabel='Avaliação Atribuída', ylabel='Quantidade de Avaliações')
plt.show()
plt.figure(figsize = (22, 8))

ax = sns.countplot(x='Classificação', data=resenha_Nubank)
ax.set(title = 'Quantidade de Cada Avaliação do Nubank', xlabel='Avaliação Atribuída', ylabel='Quantidade de Avaliações')
plt.show()
plt.figure(figsize = (22, 8))

sns.lineplot(x='Data', y='Classificação', data=resenha_Bradesco, estimator = 'mean', markers=True, color='red', label="Bradesco")
plt.show()
plt.figure(figsize = (22, 8))

sns.lineplot(x='Data', y='Classificação', data=resenha_B_Brasil, estimator = 'mean', markers=True, color='goldenrod', label="Banco do Brasil")
plt.show()
plt.figure(figsize = (22, 8))

sns.lineplot(x='Data', y='Classificação', data=resenha_Itau, estimator = 'mean', markers=True, color='coral', label='Banco Itaú')
plt.show()
plt.figure(figsize = (22, 8))

sns.lineplot(x='Data', y='Classificação', data=resenha_Caixa, estimator = 'mean', markers=True, color='blue', label='Caixa Econômica Federal')
plt.show()
plt.figure(figsize = (22, 8))

sns.lineplot(x='Data', y='Classificação', data=resenha_Nubank, estimator = 'mean', markers=True, color='purple', label='Nubank')
plt.show()
plt.figure(figsize = (22, 8))

sns.lineplot(x='Data', y='qtde_caracteres', data=resenha_Bradesco, estimator = 'mean', markers=True, color='red', label="Bradesco")
plt.show()
plt.figure(figsize = (22, 8))

sns.lineplot(x='Data', y='qtde_caracteres', data=resenha_B_Brasil, estimator = 'mean', markers=True, color='goldenrod', label="Banco do Brasil")
plt.show()
plt.figure(figsize = (22, 8))

sns.lineplot(x='Data', y='qtde_caracteres', data=resenha_Itau, estimator = 'mean', markers=True, color='coral', label='Banco Itaú')
plt.show()
plt.figure(figsize = (22, 8))

sns.lineplot(x='Data', y='qtde_caracteres', data=resenha_Caixa, estimator = 'mean', markers=True, color='blue', label='Caixa Econômica Federal')
plt.show()
plt.figure(figsize = (22, 8))

sns.lineplot(x='Data', y='qtde_caracteres', data=resenha_Nubank, estimator = 'mean', markers=True, color='purple', label='Nubank')
plt.show()
plt.figure(figsize = (22, 8))

sns.distplot(resenhas_df.qtde_caracteres.dropna(), kde=False).set_title('Distribuição da Quantidade de Caracteres')
plt.show()
plt.figure(figsize = (22, 8))

sns.distplot(resenhas_df.qtde_palavras.dropna(), kde=False).set_title('Distribuição da Quantidade de Palavras')
plt.show()
g = sns.FacetGrid(resenhas_df.dropna(), row='Instituição', height=4.5, aspect=4)
g = g.map(plt.hist, 'qtde_caracteres', bins=40)
g = sns.FacetGrid(resenhas_df.dropna(), row='Instituição', height=4.5, aspect=4)
g = g.map(plt.hist, 'qtde_palavras', bins=40)
plt.figure(figsize = (22, 8))

ax = sns.scatterplot(x='qtde_palavras', y='Classificação', data=resenhas_df)
plt.figure(figsize = (22, 8))

ax = sns.scatterplot(x='qtde_caracteres', y='Classificação', data=resenhas_df)
plt.figure(figsize = (22, 8))

ax = sns.countplot(y='Elogio quanto ao app', hue='Instituição', data=resenhas_df)
ax.set(title = 'Elogio quanto ao app por instituição', xlabel='Quantidade de atribuições', ylabel='Elogio quanto ao app')
plt.show()
plt.figure(figsize = (22, 8))

ax = sns.countplot(y='Elogio a Instituição', hue='Instituição', data=resenhas_df)
ax.set(title = 'Elogio a Instituição por instituição', xlabel='Quantidade de atribuições', ylabel='Elogio à Instituição')
plt.show()
plt.figure(figsize = (22, 8))

ax = sns.countplot(y='Reclamação quanto ao app', hue='Instituição', data=resenhas_df)
ax.set(title = 'Reclamação quanto ao app por instituição', xlabel='Quantidade de atribuições', ylabel='Reclamação quanto ao app')
plt.show()
plt.figure(figsize = (22, 8))

ax = sns.countplot(y='Reclamação a Instituição', hue='Instituição', data=resenhas_df)
ax.set(title = 'Reclamação a Instituição por instituição', xlabel='Quantidade de atribuições', ylabel='Reclamação qanto a Instituição')
plt.show()
plt.figure(figsize = (22, 8))

ax = sns.countplot(y='Não Classificável', hue='Instituição', data=resenhas_df)
ax.set(title = 'Resenha não classificável por instituição', xlabel='Quantidade de atribuições', ylabel='Comentário não Classificável')
plt.show()
