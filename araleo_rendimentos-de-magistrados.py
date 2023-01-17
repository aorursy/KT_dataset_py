import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



df = pd.read_csv("../input/salaries-of-magistrates-brasil/contracheque.csv")
df.info()
df.head()
print(f"total linhas: {df.shape[0]}, total NaNs CPF: {df['cpf'].isnull().sum()}")
df = df.drop(labels=["cpf", "url"], axis=1)
df.describe()
print(list(df.rendimento_liquido.sort_values(ascending=False).head(20)))

df.rendimento_liquido.sort_values(ascending=False).head(20).plot.bar();
df_idx_nomes = df.set_index("cargo", drop=True)

maiores_rendimentos = df_idx_nomes.rendimento_liquido.sort_values(ascending=False)[2:12]

maiores_rendimentos = maiores_rendimentos.append(pd.Series(1000.00, index=["Salário Mínimo"]))

ax = maiores_rendimentos.plot.bar(figsize=(20, 12), rot=30, title="Maiores rendimentos de magistrados x Salário mínimo - em R$ - nov/17 a abr/18")

ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()]);
df_maiores = df[df["rendimento_liquido"].isin(maiores_rendimentos)].sort_values(by="rendimento_liquido", ascending=False)

df_maiores = df_maiores[["nome", "cargo", "subsidio", "direitos_pessoais", "indenizacoes", "direitos_eventuais", "total_de_rendimentos", "previdencia_publica", "imposto_de_renda", "total_de_descontos", "rendimento_liquido", "tribunal", "mesano_de_referencia"]]

df_stacked = df_maiores[["cargo", "subsidio", "direitos_pessoais", "indenizacoes", "direitos_eventuais"]]

df_stacked = df_stacked.set_index("cargo", drop=True)

ax = df_stacked.plot.bar(figsize=(20, 8), stacked=True, title="Composição dos maiores rendimentos de magistrados antes dos descontos - nov/17 a abr/18", rot=30)

ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()]);
df_maior_cem_mil = df[df.rendimento_liquido > 100000]

df_menor_zero = df[df.rendimento_liquido < 0]

print(f"maiores cem mil: {len(df_maior_cem_mil) / len(df):.5f}")

print(f"menores zero: {len(df_menor_zero) / len(df):.5f}")
df_nomes = df[(df.rendimento_liquido > 0) & (df.rendimento_liquido < 100000)]
df_max_nomes = df_nomes["rendimento_liquido"].groupby(df_nomes.nome).max()

ax = df_max_nomes.plot.box(figsize=(10, 5), title="Box plot do maior rendimento de cada magistrado entre nov/17 e abr/18")

ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()]);
df_medias_nomes = df_nomes["rendimento_liquido"].groupby(df_nomes.nome).mean()

ax = df_medias_nomes.plot.box(figsize=(10, 5), title="Box plot da média de rendimentos de cada magistrado entre nov/17 e abr/18")

ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()]);
media_magistrados = df["rendimento_liquido"].groupby(df.nome).mean().mean()

salario_minimo = 1000.00

media_nacional = 2100.00

series_salarios = pd.Series([media_magistrados, media_nacional, salario_minimo], index=["Média magistrados", "Média Nacional", "Salário Mínimo"])

ax = series_salarios.plot.bar(figsize=(8,6), rot=0, title="Média salarial em R$")

ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()]);
maiores_diarias = df.diarias.sort_values(ascending=False)[:10]



print(list(maiores_diarias))

ax = maiores_diarias.plot.bar(figsize=(8,6), title="Maiores valores recebidos a título de diárias por magistrado - nov/17 a abr/18")

ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()]);
df_diarias = df[["diarias", "mesano_de_referencia"]].groupby("mesano_de_referencia").sum()

ax = df_diarias.plot(figsize=(8, 6), rot=30, legend=False, title="Valor total de diárias pagas a magistrados em R$ - nov/17 a abr/18")

ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()]);
regioes = {

    "sudeste": ("Minas Gerais", "São Paulo", "Rio de Janeiro", "Espírito Santo"),

    "sul": ("Rio Grande do Sul", "Paraná", "Santa Catarina"),

    "centro_oeste": ("Goiás", "Mato Grosso", "Mato Grosso do Sul", "Distrito Federal"),

    "nordeste": ("Alagoas", "Bahia", "Ceará", "Maranhão", "Paraíba", "Pernambuco", "Piauí", "Rio Grande do Norte", "Sergipe"),

    "norte": ("Acre", "Amapá", "Amazonas", "Pará", "Rondônia", "Roraima", "Tocantins")  

}



df_tjs = df[df.tribunal.str.startswith("Tribunal de Justiça")]

df_trfs = df[df.tribunal.str.startswith("Tribunal Regional Federal")]

df_trts = df[df.tribunal.str.startswith("Tribunal Regional do Trabalho")]

df_tres = df[df.tribunal.str.startswith("Tribunal Regional Eleitoral")]
series_regioes = {}

for regiao in regioes:

    _df = df_tjs[df_tjs.tribunal.str.endswith(regioes[regiao])].groupby(df_tjs.tribunal).mean()

    series_regioes[regiao] = _df[["rendimento_liquido"]].mean().sort_values()

df_regioes = pd.DataFrame.from_dict(series_regioes, orient="index").sort_values(by="rendimento_liquido", ascending=False)

ax = df_regioes.plot.bar(figsize=(10, 6), rot=0, title="Média de rendimento dos magistrados de Tribunais de Justiça entre nov/17 e abr/18 por região", legend=False)

ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()]);
reg = "centro_oeste"

tjs_sudeste = df_tjs[df_tjs.tribunal.str.endswith(regioes[reg])].groupby(df_tjs.tribunal).mean()

tjs_sudeste = tjs_sudeste[["rendimento_liquido"]].sort_values(by="rendimento_liquido", ascending=False)

ax = tjs_sudeste.plot.bar(figsize=(12,6), title=f"Média de rendimentos dos magistrados da região {reg} entre nov/17 e abr/18 em R$", rot=30, legend=False)

ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()]);
trfs_data = df_trfs.groupby(df_trfs.tribunal).mean()

trfs_data = trfs_data[["rendimento_liquido"]].sort_values(by="rendimento_liquido", ascending=False)

ax = trfs_data.plot.bar(figsize=(12,6), title="Média de rendimentos dos magistrados de TRFs entre nov/17 e abr/18 em R$", rot=30)

ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()]);
media_tjs = df_regioes.values.mean()

media_trfs = trfs_data.values.mean()



df_stj = df[df.tribunal == "Superior Tribunal de Justiça"]

media_stj = df_stj.rendimento_liquido.values.mean()



series_medias = pd.Series([media_tjs, media_trfs, media_stj], index=["TJs", "TRFs", "STJ"])

ax = series_medias.plot.bar(figsize=(8,6), rot=0, title="Média de rendimentos dos magistrados em R$")

ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()]);