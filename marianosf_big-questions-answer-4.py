import pandas as pd
%matplotlib inline
deputados = pd.read_csv('../input/deputies_1998_2018.csv')
proposicoes = pd.read_json('../input/propositions_det_1998_2018.json')
idPEC = [136]
# Filtrando por PEC
proposicoes = proposicoes[proposicoes['dados.idTipo'] == idPEC]
# Retirando URI vazias
proposicoes = proposicoes[proposicoes['dados.uriAutores'].notnull()]
# Retirando deputados None
deputados = deputados[deputados['dados.id'].notnull()]
proposicoes['autoresID'] = proposicoes['dados.uriAutores'].apply(lambda uri: uri[uri.rindex('/') + 1:]).astype('int32')
deputados['autoresID'] = deputados['dados.id'].astype('int32')
tops = deputados.set_index('autoresID').join(proposicoes['autoresID'].value_counts())
tops[['dados.nomeCivil','autoresID']].sort_values(by='autoresID',ascending=False).head(20)
tops[['dados.nomeCivil','autoresID']].hist(bins=40,figsize=(9,9))
tops[['dados.nomeCivil','autoresID']].boxplot(vert=False,figsize=(25,5))