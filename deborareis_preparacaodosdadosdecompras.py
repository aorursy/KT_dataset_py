# Importa as bibliotecas

import pandas as pd

import glob

path = '../input'
# Item Licitacao

arquivos = glob.glob(path + "/2018??_ItemLicitao.csv")

lista = []

for i in arquivos:

    dados = pd.read_csv(i, delimiter=';', encoding='latin1',

                        names=["codigoOrgao", "nomeOrgao", "codigoUG", "nomeUG", 

                               "numeroLicitacao", "cnpjVencedor", "nomeVencedor", 

                               "numeroItem", "descricao", "qtdItem", "valorItem"], 

                        dtype={'codigoOrgao':object, "codigoUG":object,"numeroLicitacao":object,

                               "cnpjVencedor":object,"numeroItem":object},

                        header=0)

    lista.append(dados)

item = pd.concat(lista, axis=0, ignore_index=True)

item.to_csv("itemLicitacao.csv", index=False)
# Licitacao

arquivos = glob.glob(path + "/2018??_Licitao.csv")

lista = []

for i in arquivos:

    dados = pd.read_csv(i, delimiter=';', encoding='latin1',

                        names=["numLicitacao", "processo", "objeto", "modalidade", "situacao", "codigoOrgaoSuperior", 

                               "nomeOrgaoSuperior", "codigoOrgao", "nomeOrgao", "codigoUG", "nomeUG",

                               "municipio","dataPublicacao","dataAbertura","valorLicitacao"], 

                        dtype={"numLicitacao":object, "processo":object, "codigoOrgaoSuperior":object,

                               "codigoOrgao":object,"codigoUG":object},

                        header=0)

    lista.append(dados)

licitacao = pd.concat(lista, axis=0, ignore_index=True)

licitacao.to_csv("licitacao.csv", index=False)
# Participantes Licitacao

arquivos = glob.glob(path + "/2018??_ParticipantesLicitao.csv")

lista = []

for i in arquivos:

    dados = pd.read_csv(i, delimiter=';', encoding='latin1',

                        names=["codigoOrgao", "nomeOrgao", "codigoUG", "nomeUG", "numeroLicitacao","codigoItemCompra",

                               "descricaoItemCompra","cnpjParticipante","nomeParticipante","flagVencedor"], 

                        dtype={'codigoOrgao':object, "codigoUG":object,"numeroLicitacao":object,

                               "codigoItemCompra":object,"cnpjParticipante":object},

                        header=0)

    lista.append(dados)

participante = pd.concat(lista, axis=0, ignore_index=True)

participante.to_csv("participantesLicitacao.csv", index=False)
# Modalidade

modalidade = pd.read_csv('../input/D_CMPR_MODALIDADE_COMPRA_201908051050.csv', sep=";")

modalidade = pd.DataFrame(modalidade, columns=['ID_CMPR_MODALIDADE_COMPRA','DS_CMPR_MODALIDADE_COMPRA'])

modalidade = modalidade.rename(columns={'ID_CMPR_MODALIDADE_COMPRA':'codmodalidade','DS_CMPR_MODALIDADE_COMPRA':'modalidade'})

modalidade.to_csv('modalidade.csv', index=False)