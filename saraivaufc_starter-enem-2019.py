import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_path = "/kaggle/input/DADOS/MICRODADOS_ENEM_2019.csv"

df = pd.read_csv(data_path, sep=';', encoding = "ISO-8859-1", chunksize=1000000)

df = pd.concat(df, ignore_index=True)

df.head()
def build_table(df, index, colunms):
    df2 = df[index.keys()]

    pieces = []
    for col in df2.columns:
        tmp_series = df2[col].value_counts()
        tmp_series.name = col
        pieces.append(tmp_series)
    df_value_counts = pd.concat(pieces, axis=1)

    df_t = df_value_counts.T
    df_t.rename(columns=colunms, 
                index=index, inplace=True)

    return df_t


def annotate(ax, labels=None):
    if labels: 
        ax.set_xticklabels(labels)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()))
    
    return ax
build_table(df, {'TP_SEXO': 'Sexo'}, {'M': 'Masculino', 'F': 'Feminino'})
ax = df['TP_SEXO'].value_counts(sort=False).plot(title='Sexo', kind='bar', rot=0)

annotate(ax)
df['SG_UF_RESIDENCIA'].value_counts().plot(title='Unidade da Federação de residência',kind='bar', figsize=(20,5), rot=0)
df['NU_IDADE'].value_counts(sort=False).plot(title='Idade',kind='bar', figsize=(20,5))
ax = df['TP_SEXO'].value_counts(sort=False).plot(title='Sexo', kind='bar', rot=0)

annotate(ax)
ax = df['TP_ESTADO_CIVIL'].value_counts(sort=False).plot(title='Estado Civil',kind='bar', figsize=(20,5), rot=0)

annotate(ax, ['Não informado', 'Solteiro(a)', 'Casado(a)/Mora com companheiro(a)', 'Divorciado(a)/Desquitado(a)/Separado(a)', 'Viúvo(a)'])
ax = df['TP_COR_RACA'].value_counts(sort=False).plot(title='Cor/raça', kind='bar', figsize=(20,5), rot=0)

annotate(ax, ['Não declarado','Branca','Preta','Parda','Amarela','Indígena'])
ax = df['TP_NACIONALIDADE'].value_counts(sort=False).plot(title='Nacionalidade', kind='bar', figsize=(20,5), rot=0)

annotate(ax, ['Não informado', 'Brasileiro(a)', 'Brasileiro(a) Naturalizado(a)', 'Estrangeiro(a)', 'Brasileiro(a) Nato(a), nascido(a) no exterior'])
ax = df['TP_ST_CONCLUSAO'].value_counts(sort=False).plot(title='Situação de conclusão do Ensino Médio', kind='bar', figsize=(20,5), rot=0)

annotate(ax, ['Já concluí o Ensino Médio', 'Estou cursando e concluirei o Ensino Médio em 2019', 'Estou cursando e concluirei o Ensino Médio após 2019', 'Não concluí e não estou cursando o Ensino Médio'])
ax = df['TP_ESCOLA'].value_counts(sort=False).plot(title='Tipo de escola do Ensino Médio',kind='bar', figsize=(20,5), rot=0)

annotate(ax, ['Não Respondeu', 'Pública', 'Privada', 'Exterior'])
ax = df['TP_ANO_CONCLUIU'].value_counts(sort=False).plot(title='Ano de Conclusão do Ensino Médio', kind='bar', figsize=(20,5), rot=0)

annotate(ax, ['Não informado', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008', '2007', 'Antes de 2007'])
ax = df['IN_TREINEIRO'].value_counts(sort=False).plot(title='Indica se o inscrito fez a prova com intuito de apenas treinar seus conhecimentos', kind='bar', figsize=(20,5), rot=0)

annotate(ax, ['Não', 'Sim'])
ax = df['TP_DEPENDENCIA_ADM_ESC'].value_counts(sort=False).plot(title='Dependência administrativa (Escola)', kind='bar', figsize=(20,5), rot=0)

annotate(ax, ['Estadual','Municipal','Privada','Federal'])
ax = df['TP_LOCALIZACAO_ESC'].value_counts(sort=False).plot(title='Localização (Escola)', kind='bar', figsize=(20,5), rot=0)

annotate(ax, ['Rural', 'Urbana'])
index = {
    'IN_BAIXA_VISAO' : 'Indicador de baixa visão',
    'IN_CEGUEIRA' : 'Indicador de cegueira',
    'IN_SURDEZ' : 'Indicador de surdez',
    'IN_SURDO_CEGUEIRA' : 'Indicador de surdo-cegueira',
    'IN_DEFICIENCIA_FISICA' : 'Indicador de deficiência física',
    'IN_DEFICIENCIA_MENTAL' : 'Indicador de deficiência mental',
    'IN_DEFICIT_ATENCAO' : 'Indicador de déficit de atenção',
    'IN_DISLEXIA' : 'Indicador de dislexia',
    'IN_DISCALCULIA' : 'Indicador de discalculia',
    'IN_AUTISMO' : 'Indicador de autismo',
    'IN_VISAO_MONOCULAR' : 'Indicador de visão monocular',
    'IN_OUTRA_DEF' : 'Indicador de outra deficiência ou condição especial'
}

build_table(df, index, {0: 'Não', 1: 'Sim'})
index = {
    'IN_GESTANTE': 'Indicador de gestante',
    'IN_LACTANTE': 'Indicador de lactante',
    'IN_IDOSO': 'Indicador de inscrito idoso',
    'IN_ESTUDA_CLASSE_HOSPITALAR': 'Indicador de inscrição em Unidade Hospitalar'
}

build_table(df, index, {0: 'Não', 1: 'Sim'})
index = {
    'IN_SEM_RECURSO':'Indicador de inscrito que não requisitou nenhum recurso',
    'IN_BRAILLE':'Indicador de solicitação de prova em braille',
    'IN_AMPLIADA_24':'Indicador de solicitação de prova superampliada com fonte tamanho 24',
    'IN_AMPLIADA_18':'Indicador de solicitação de prova ampliada com fonte tamanho 18',
    'IN_LEDOR':'Indicador de solicitação de auxílio para leitura (ledor)',
    'IN_ACESSO':'Indicador de solicitação de sala de fácil acesso',
    'IN_TRANSCRICAO':'Indicador de solicitação de auxílio para transcrição',
    'IN_LIBRAS':'Indicador de solicitação de Tradutor- Intérprete Libras',
    'IN_TEMPO_ADICIONAL':'Indicador de solicitação de tempo adicional',
    'IN_LEITURA_LABIAL':'Indicador de solicitação de leitura labial',
    'IN_MESA_CADEIRA_RODAS':'Indicador de solicitação de mesa para cadeira de rodas',
    'IN_MESA_CADEIRA_SEPARADA':'Indicador de solicitação de mesa e cadeira separada',
    'IN_APOIO_PERNA':'Indicador de solicitação de apoio de perna e pé',
    'IN_GUIA_INTERPRETE':'Indicador de solicitação de guia intérprete',
    'IN_COMPUTADOR':'Indicador de solicitação de computador',
    'IN_CADEIRA_ESPECIAL':'Indicador de solicitação de cadeira especial',
    'IN_CADEIRA_CANHOTO':'Indicador de solicitação de cadeira para canhoto',
    'IN_CADEIRA_ACOLCHOADA':'Indicador de solicitação de cadeira acolchoada',
    'IN_PROVA_DEITADO':'Indicador de solicitação para fazer prova deitado em maca ou mobiliário similar',
    'IN_MOBILIARIO_OBESO':'Indicador de solicitação de mobiliário adequado para obeso',
    'IN_LAMINA_OVERLAY':'Indicador de solicitação de lâmina overlay',
    'IN_PROTETOR_AURICULAR':'Indicador de solicitação de protetor auricular',
    'IN_MEDIDOR_GLICOSE':'Indicador de solicitação de medidor de glicose e/ou aplicação de insulina',
    'IN_MAQUINA_BRAILE':'Indicador de solicitação de máquina Braile e/ou Reglete e Punção',
    'IN_SOROBAN':'Indicador de solicitação de soroban',
    'IN_MARCA_PASSO':'Indicador de solicitação de marca-passo (impeditivo de uso de detector de metais)',
    'IN_SONDA':'Indicador de solicitação de sonda com troca periódica',
    'IN_MEDICAMENTOS':'Indicador de solicitação de medicamentos',
    'IN_SALA_INDIVIDUAL':'Indicador de solicitação de sala especial individual',
    'IN_SALA_ESPECIAL':'Indicador de solicitação de sala especial até 20 participantes',
    'IN_SALA_ACOMPANHANTE':'Indicador de solicitação de sala reservada para acompanhantes',
    'IN_MOBILIARIO_ESPECIFICO':'Indicador de solicitação de mobiliário específico',
    'IN_MATERIAL_ESPECIFICO':'Indicador de solicitação de material específico',
    'IN_NOME_SOCIAL':'Indicador de inscrito que se declarou travesti, transexual ou transgênero e solicitou atendimento pelo Nome Social, conforme é reconhecido socialmente em consonância com sua identidade de gênero'
}

build_table(df, index, {0: 'Não', 1: 'Sim'})
index = {
    'TP_PRESENCA_CN': 'Presença na prova objetiva de Ciências da Natureza',
    'TP_PRESENCA_CH': 'Presença na prova objetiva de Ciências Humanas',
    'TP_PRESENCA_LC': 'Presença na prova objetiva de Linguagens e Códigos',
    'TP_PRESENCA_MT': 'Presença na prova objetiva de Matemática'
}

build_table(df, index, {0: 'Faltou à prova', 1: 'Presente na prova', 2: 'Eliminado na prova'})
ax = df['CO_PROVA_CN'].value_counts(sort=False).plot(title='Código do tipo de prova de Ciências da Natureza', kind='bar', figsize=(25,5), rot=0)

annotate(ax, ['Azul','Amarela','Cinza','Rosa','Laranja - Adaptada Ledor','Verde - Videoprova - Libras','Amarela (Reaplicação)','Cinza (Reaplicação)','Azul (Reaplicação)','Rosa (Reaplicação)'])
ax = df['CO_PROVA_CH'].value_counts(sort=False).plot(title='Código do tipo de prova de Ciências Humanas', kind='bar', figsize=(25,5), rot=0)

annotate(ax, ['Azul', 'Amarela', 'Branca', 'Rosa', 'Laranja - Adaptada Ledor', 'Verde - Videoprova - Libras', 'Azul (Reaplicação)', 'Amarelo (Reaplicação)', 'Branco (Reaplicação)', 'Rosa (Reaplicação)', 'Laranja - Adaptada Ledor (Reaplicação)'])
ax = df['CO_PROVA_LC'].value_counts(sort=False).plot(title='Código do tipo de prova de Linguagens e Códigos', kind='bar', figsize=(25,5), rot=0)

annotate(ax, ['Azul', 'Amarela', 'Branca', 'Rosa', 'Laranja - Adaptada Ledor', 'Verde - Videoprova - Libras', 'Azul (Reaplicação)', 'Amarelo (Reaplicação)', 'Branco (Reaplicação)', 'Rosa (Reaplicação)', 'Laranja - Adaptada Ledor (Reaplicação)'])
ax = df['CO_PROVA_MT'].value_counts(sort=False).plot(title='Código do tipo de prova de Matemática', kind='bar', figsize=(25,5), rot=0)

annotate(ax, ['Azul','Amarela','Cinza','Rosa','Laranja - Adaptada Ledor','Verde - Videoprova - Libras','Amarela (Reaplicação)','Cinza (Reaplicação)','Azul (Reaplicação)','Rosa (Reaplicação)'])
ax = df['NU_NOTA_CN'].plot.hist(title='Nota da prova de Ciências da Natureza',bins=20, alpha=0.5, figsize=(25,5))

annotate(ax)
ax = df['NU_NOTA_CH'].plot.hist(title='Nota da prova de Ciências Humanas',bins=20, alpha=0.5, figsize=(25,5))

annotate(ax)
ax = df['NU_NOTA_LC'].plot.hist(title='Nota da prova de Linguagens e Códigos',bins=20, alpha=0.5, figsize=(25,5))

annotate(ax)
ax = df['NU_NOTA_MT'].plot.hist(title='Nota da prova de Matemática',bins=20, alpha=0.5, figsize=(25,5))

annotate(ax)
ax = df['TP_LINGUA'].value_counts(sort=False).plot(title='Língua Estrangeira ', kind='bar', figsize=(25,5), rot=0)

annotate(ax, ['Inglês', 'Espanhol'])
ax = df['TP_STATUS_REDACAO'].value_counts(sort=False).plot(title='Situação da redação do participante', kind='bar', figsize=(25,5), rot=0)

annotate(ax, ['Anulada', 'Cópia Texto Motivador', 'Em Branco', 'Fuga ao tema', 'Não atendimento ao tipo textual', 'Texto insuficiente', 'Parte desconectada', 'Sem problemas'])
ax = df['NU_NOTA_COMP1'].plot.hist(title='Nota da competência 1 - Demonstrar domínio da modalidade escrita formal da Língua Portuguesa.',bins=20, alpha=0.5, figsize=(25,5))

annotate(ax)
ax = df['NU_NOTA_COMP2'].plot.hist(title='Nota da competência 2 - Compreender a proposta de redação e aplicar conceitos das várias áreas de conhecimento para desenvolver o tema, dentro dos limites estruturais do texto dissertativo-argumentativo em prosa.',bins=20, alpha=0.5, figsize=(25,5))

annotate(ax)
ax = df['NU_NOTA_COMP3'].plot.hist(title='Nota da competência 3 - Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de vista.',bins=20, alpha=0.5, figsize=(25,5))

annotate(ax)
ax = df['NU_NOTA_COMP4'].plot.hist(title='Nota da competência 4 - Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação.',bins=20, alpha=0.5, figsize=(25,5))

annotate(ax)
ax = df['NU_NOTA_COMP5'].plot.hist(title='Nota da competência 5 - Elaborar proposta de intervenção para o problema abordado, respeitando os direitos humanos.',bins=20, alpha=0.5, figsize=(25,5))

annotate(ax)
ax = df['NU_NOTA_REDACAO'].plot.hist(title='Nota da prova de redação',bins=20, alpha=0.5, figsize=(25,5))

annotate(ax)
ax = df['Q001'].value_counts(sort=True).plot.barh(title='Até que série seu pai, ou o homem responsável por você, estudou?', figsize=(25,5), rot=0)

ax.set_yticklabels([
    'Completou o Ensino Médio, mas não completou a Faculdade',
    'Não completou a 4ª série/5º ano do Ensino Fundamental',
    'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental',
    'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio',
    'Não sei',
    'Completou a Faculdade, mas não completou a Pós-graduação',
    'Nunca estudou',
    'Completou a Pós-graduação'
])

annotate(ax)
ax = df['Q002'].value_counts(sort=True).plot.barh(title='Até que série sua mãe, ou a mulher responsável por você, estudou?', figsize=(25,5), rot=0)

ax.set_yticklabels([
    'Completou o Ensino Médio, mas não completou a Faculdade',
    'Não completou a 4ª série/5º ano do Ensino Fundamental',
    'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental',
    'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio',
    'Completou a Faculdade, mas não completou a Pós-graduação',
    'Completou a Pós-graduação',
    'Nunca estudou',
    'Não sei'
])

annotate(ax)
ax = df['Q005'].value_counts(sort=False).plot(title='Incluindo você, quantas pessoas moram atualmente em sua residência?', kind='bar', figsize=(25,5), rot=0)

annotate(ax)
labels = ['Não.', 'Sim, uma.', 'Sim, duas.', 'Sim, três.', 'Sim, quatro ou mais.']
ax = df['Q008'].value_counts().sort_index().plot(title='Na sua residência tem banheiro?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, labels)
ax = df['Q009'].value_counts().sort_index().plot(title='Na sua residência tem quartos para dormir?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, labels)
ax = df['Q010'].value_counts().sort_index().plot(title='Na sua residência tem carro?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, labels)
ax = df['Q011'].value_counts().sort_index().plot(title='Na sua residência tem motocicleta?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, labels)
ax = df['Q012'].value_counts().sort_index().plot(title='Na sua residência tem geladeira?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, labels)
ax = df['Q013'].value_counts().sort_index().plot(title='Na sua residência tem freezer (independente ou segunda porta da geladeira)?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, labels)
ax = df['Q014'].value_counts().sort_index().plot(title='Na sua residência tem máquina de lavar roupa? (o tanquinho NÃO deve ser considerado)', kind='bar', figsize=(25,5), rot=0)

annotate(ax, labels)
ax = df['Q015'].value_counts().sort_index().plot(title='Na sua residência tem máquina de secar roupa (independente ou em conjunto com a máquina de lavar roupa)?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, labels)
ax = df['Q016'].value_counts().sort_index().plot(title='Na sua residência tem forno micro-ondas?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, labels)
ax = df['Q017'].value_counts().sort_index().plot(title='Na sua residência tem máquina de lavar louça?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, labels)
ax = df['Q018'].value_counts().sort_index().plot(title='Na sua residência tem aspirador de pó?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, ['Não', 'Sim'])
ax = df['Q019'].value_counts().sort_index().plot(title='Na sua residência tem televisão em cores?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, labels)
ax = df['Q020'].value_counts().sort_index().plot(title='Na sua residência tem aparelho de DVD?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, ['Não', 'Sim'])
ax = df['Q021'].value_counts().sort_index().plot(title='Na sua residência tem TV por assinatura?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, ['Não', 'Sim'])
ax = df['Q022'].value_counts().sort_index().plot(title='Na sua residência tem telefone celular?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, labels)
ax = df['Q023'].value_counts().sort_index().plot(title='Na sua residência tem telefone fixo?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, ['Não', 'Sim'])
ax = df['Q024'].value_counts().sort_index().plot(title='Na sua residência tem computador?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, labels)
ax = df['Q025'].value_counts().sort_index().plot(title='Na sua residência tem acesso à Internet?', kind='bar', figsize=(25,5), rot=0)

annotate(ax, ['Não', 'Sim'])