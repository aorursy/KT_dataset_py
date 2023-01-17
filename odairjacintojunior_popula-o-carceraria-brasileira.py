import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import warnings

import xgboost as xgb

import xgboost

import statsmodels.formula.api as smf

import shap

import statsmodels.api as sm

import sys

import itertools

import plotly.figure_factory as ff

import plotly

import plotly.graph_objs as go

import folium



from scipy import stats

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.ensemble import RandomForestClassifier

from IPython.display import Image

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn import metrics

from io import StringIO

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from IPython.display import display, Markdown

from plotly.offline import *

from sklearn.tree import DecisionTreeClassifier, export_graphviz



%matplotlib inline 

warnings.filterwarnings("ignore") 

# print the JS visualization code to the notebook

shap.initjs()
! pip install pydot==1.4.1 xgboost==0.90 shap==0.30.1 #shap==0.29.3

! pip install folium

! pip install cufflinks plotly

! pip install plotly
df_2014 = pd.read_excel('/kaggle/input/base2014/base-de-dados-infopen-2014.xlsx')
# Aumentando as configurações de visualizações para as colunas pelo pandas. 

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 2000)

pd.set_option('display.width', 2000)
df_2014.head(1)
df_2014.shape
df_2014.rename(columns={'Invite: unidade': 'Unidade_Prisional',

                        'Invite: UF': 'UF_Entrada',

                        'Nome da unidade prisional:': 'Nome_UP',

                        'Nome do responsável pelo preenchimento:':'Resp_Pelo_Prrenchimento',

                        'Endereço da Unidade:':'Endereco_Unidade',

                        'CEP:':'CEP',

                        'UF:':'UF',

                        'Telefone para população obter informações sobre visitação:':'Tel_Info_Para_Pop',

                        '1.1. Estabelecimento originalmente destinado a pessoa privadas de liberdade do sexo:' :'UP_Dest_M',

                        '1.2. Tipo de estabelecimento - originalmente destinado:': 'UP_Org_Dest',

                        '1.2. Tipo de estabelecimento - originalmente destinado: [other]':'UP_Org_Dest_Outros',

                        '1.3. Capacidade do estabelecimento: | vagas - presos provisórios | Masculino': 'Capacidade_Prov_M',

                        '1.3. Capacidade do estabelecimento: | vagas - presos provisórios | Feminino' : 'Capacidade_Prov_F',

                        '1.3. Capacidade do estabelecimento: | vagas - regime fechado | Masculino': 'Capacidade_Fechado_M',

                        '1.3. Capacidade do estabelecimento: | vagas - regime fechado | Feminino' : 'Capacidade_Fechado_F',

                        '1.3. Capacidade do estabelecimento: | vagas - regime semiaberto | Masculino': 'Capacidade_Semi_M',

                        '1.3. Capacidade do estabelecimento: | vagas - regime semiaberto | Feminino': 'Capacidade_Semi_F',

                        '1.3. Capacidade do estabelecimento: | vagas - regime aberto | Masculino': 'Capacidade_Aberto_M',

                        '1.3. Capacidade do estabelecimento: | vagas - regime aberto | Feminino':'Capacidade_Aberto_F',

                        '1.3. Capacidade do estabelecimento: | vagas - Regime Disciplinar Diferenciado (RDD) | Masculino' : 'Capacidade_RDD_M',

                        '1.3. Capacidade do estabelecimento: | vagas - Regime Disciplinar Diferenciado (RDD) | Feminino' : 'Capacidade_RDD_F',

                        '1.3. Capacidade do estabelecimento: | vagas - Medidas de segurança de internação | Masculino': 'Capacidade_Intern_M',

                        '1.3. Capacidade do estabelecimento: | vagas - Medidas de segurança de internação | Feminino': 'Capacidade_Intern_F',

                        '1.3. Capacidade do estabelecimento: | vagas - Outro(s). Qual(is)? (especificar abaixo) | Masculino': 'Capacidade_Outros_M',

                        '1.3. Capacidade do estabelecimento: | vagas - Outro(s). Qual(is)? (especificar abaixo) | Feminino': 'Capacidade_Outros_F',

                        '1.3.a. Especifique o(s) outro(s) regimes(s) citados acima:': 'Capacidade_Outros',

                        '1.3.1. Capacidade do estabelecimento: | Celas interditadas/desativadas e respectivas vagas | Quantidade de celas não aptas': 'Celas_N_Aptas',

                        '1.3.1. Capacidade do estabelecimento: | Celas interditadas/desativadas e respectivas vagas | Vagas desativadas | Masculino': 'Celas_N_Aptas_M',

                        '1.3.1. Capacidade do estabelecimento: | Celas interditadas/desativadas e respectivas vagas | Vagas desativadas | Feminino': 'Celas_N_Aptas_F',

                        '1.4. Gestão do estabelecimento:': 'Tipo_Gestao',

                        '1.5. Quais serviços são terceirizados? (marcar mais de uma resposta, se aplicável) [Nenhum]' : 'Existe_Servico_Terc',

                        '1.5. Quais serviços são terceirizados? (marcar mais de uma resposta, se aplicável) [Alimentação]': 'Alimentacao_Terc',

                        '1.5. Quais serviços são terceirizados? (marcar mais de uma resposta, se aplicável) [Limpeza]': 'Limpeza_Terc',

                        '1.5. Quais serviços são terceirizados? (marcar mais de uma resposta, se aplicável) [Lavanderia]': 'Lavanderia_Terc',

                        '1.5. Quais serviços são terceirizados? (marcar mais de uma resposta, se aplicável) [Saúde]': 'Saude_Terc',

                        '1.5. Quais serviços são terceirizados? (marcar mais de uma resposta, se aplicável) [Segurança]': 'Seguranca_Terc',

                        '1.5. Quais serviços são terceirizados? (marcar mais de uma resposta, se aplicável) [Assistência educacional]': 'Assist_Educ_Terc',

                        '1.5. Quais serviços são terceirizados? (marcar mais de uma resposta, se aplicável) [Assistência laboral (Exemplo: terapeuta ocupacional, instrutor, coordenador de trabalho que acompanham as atividades oferecidas na Unidade.)]':'Assist_Laboral_Terc', 

                        '1.5. Quais serviços são terceirizados? (marcar mais de uma resposta, se aplicável) [Assistência social]': 'Assist_Social_Terc',

                        '1.5. Quais serviços são terceirizados? (marcar mais de uma resposta, se aplicável) [Assistência jurídica]': 'Assist_Jur_Terc',

                        '1.5. Quais serviços são terceirizados? (marcar mais de uma resposta, se aplicável) [Serviços administrativos]': 'Serv_Adm_Terc',

                        '1.5. Quais serviços são terceirizados? (marcar mais de uma resposta, se aplicável) [Outro. Qual?]': 'Serv_Terc_Outros',

                        '1.5. Quais serviços são terceirizados? (marcar mais de uma resposta, se aplicável) [Outro. Qual?] [text]': 'Serv_Terc_Outros_Desc',

                        '1.6. Data de inauguração do estabelecimento:': 'UP_Inauguracao',

                        '1.7. O estabelecimento foi concebido como estabelecimento penal ou foi construído para outra utilização e foi adaptado?': 'Concebido_Adaptado',

                        '1.8. Possui regimento interno?': 'Regime_Interno',

                        '1.9. O regimento interno é específico para este estabelecimento ou se aplica aos demais estabelecimentos do Estado?': 'Regime_Interno_Especifico',

                        '1.9. O regimento interno é específico para este estabelecimento ou se aplica aos demais estabelecimentos do Estado? [other]':'Regime_Interno_Especifico_Outros',

                        '2.1. Há cela adequada/dormitório para gestantes? (apenas para estabelecimentos com vagas para mulheres)': 'Celas_Para_Gestante',

                        '2.1.1. Quantidade de gestantes/lactantes: | Quantidade de gestantes/ parturientes': 'Qtde_Gestantes',

                        '2.1.1. Quantidade de gestantes/lactantes: | Quantidade de lactantes': 'Qtde_Lactantes',

                        '2.2. Possui berçário e/ou centro de referência materno-infantil? (apenas para estabelecimentos com vagas para mulheres)': 'Bercario',

                        '2.2. Possui berçário e/ou centro de referência materno-infantil? (apenas para estabelecimentos com vagas para mulheres) [other]': 'Bercario_Outros',

                        '2.3. Possui creche? (apenas para estabelecimentos com vagas para mulheres)': 'Creche',

                        '2.3. Possui creche? (apenas para estabelecimentos com vagas para mulheres) [other]': 'Creche_Outros',

                        '2.4. Módulo de saúde: | Consultório médico | O espaço está disponível no estabelecimento?': 'Consultorio_Med',

                        '2.4. Módulo de saúde: | Consultório médico | Quantidade': 'Qtde_Consultorio_Med',

                        '2.4. Módulo de saúde: | Consultório médico | O espaço também é destinado a outras finalidades?': 'Consultorio_Med_Outros',

                        '2.4. Módulo de saúde: | Consultório odontológico | O espaço está disponível no estabelecimento?': 'Consultorio_Odont',

                        '2.4. Módulo de saúde: | Consultório odontológico | Quantidade': 'Qtde_Consultorio_Odont',

                        '2.4. Módulo de saúde: | Consultório odontológico | O espaço também é destinado a outras finalidades?':'Consultorio_Odont_Outros',

                        '2.4. Módulo de saúde: | Sala de coleta de material para laboratório | O espaço está disponível no estabelecimento?':'Lab_Coleta',

                        '2.4. Módulo de saúde: | Sala de coleta de material para laboratório | Quantidade': 'Qtde_Lab_Coleta',

                        '2.4. Módulo de saúde: | Sala de coleta de material para laboratório | O espaço também é destinado a outras finalidades?':'Lab_Coleta_Outros',

                        '2.4. Módulo de saúde: | Sala de curativos, suturas, vacinas e posto de enfermagem | O espaço está disponível no estabelecimento?': 'Enfermaria',

                        '2.4. Módulo de saúde: | Sala de curativos, suturas, vacinas e posto de enfermagem | Quantidade': 'Qtde_Enfermaria',

                        '2.4. Módulo de saúde: | Sala de curativos, suturas, vacinas e posto de enfermagem | O espaço também é destinado a outras finalidades?':'Enfermaria_Outros',

                        '2.4. Módulo de saúde: | Cela de observação | O espaço está disponível no estabelecimento?': 'Cela_Obs',

                        '2.4. Módulo de saúde: | Cela de observação | Quantidade': 'Qtde_Cela_Obs',

                        '2.4. Módulo de saúde: | Cela de observação | O espaço também é destinado a outras finalidades?': 'Cel_Obs_Outros',

                        '2.4. Módulo de saúde: | Cela de enfermaria com solário | O espaço está disponível no estabelecimento?': 'Cel_Enferm_Solario',

                        '2.4. Módulo de saúde: | Cela de enfermaria com solário | Quantidade': 'Qtde_Cel_Enferm_Solario',

                        '2.4. Módulo de saúde: | Cela de enfermaria com solário | O espaço também é destinado a outras finalidades?': 'Cel_Enferm_Solario_Outros',

                        '2.4. Módulo de saúde: | Sanitário para pacientes | O espaço está disponível no estabelecimento?': 'Sanitario_Pacientes',

                        '2.4. Módulo de saúde: | Sanitário para pacientes | Quantidade': 'Qtde_Sanitario_Pacientes',

                        '2.4. Módulo de saúde: | Sanitário para pacientes | O espaço também é destinado a outras finalidades?': 'Sanitario_Pacientes_Outros',

                        '2.4. Módulo de saúde: | Sanitários para equipe de saúde | O espaço está disponível no estabelecimento?': 'Sanitario_Equipe_Saude',

                        '2.4. Módulo de saúde: | Sanitários para equipe de saúde | Quantidade': 'Qtde_Sanitario_Equipe_Saude',

                        '2.4. Módulo de saúde: | Sanitários para equipe de saúde | O espaço também é destinado a outras finalidades?': 'Sanitario_Equipe_Saude_Outros',

                        '2.4. Módulo de saúde: | Farmácia ou sala de estoque/ dispensação de medicamentos | O espaço está disponível no estabelecimento?': 'Farmacia',

                        '2.4. Módulo de saúde: | Farmácia ou sala de estoque/ dispensação de medicamentos | Quantidade': 'Qtde_Farmacia',

                        '2.4. Módulo de saúde: | Farmácia ou sala de estoque/ dispensação de medicamentos | O espaço também é destinado a outras finalidades?':'Farmacia_Outros',

                        '2.4. Módulo de saúde: | Central de material esterilizado/ expurgo | O espaço está disponível no estabelecimento?':'Sala_Expurgo',

                        '2.4. Módulo de saúde: | Central de material esterilizado/ expurgo | Quantidade': 'Qtde_Sala_Expurgo',

                        '2.4. Módulo de saúde: | Central de material esterilizado/ expurgo | O espaço também é destinado a outras finalidades?':'Sala_Expurgo_Outros',

                        '2.4. Módulo de saúde: | Sala de lavagem e descontaminação | O espaço está disponível no estabelecimento?':'Sala_Descontaminação',

                        '2.4. Módulo de saúde: | Sala de lavagem e descontaminação | Quantidade':'Qtde_Sala_Descontaminação',

                        '2.4. Módulo de saúde: | Sala de lavagem e descontaminação | O espaço também é destinado a outras finalidades?':'Sala_Descontaminação_Outros',

                        '2.4. Módulo de saúde: | Sala de esterilização | O espaço está disponível no estabelecimento?':'Sala_Esterilização',

                        '2.4. Módulo de saúde: | Sala de esterilização | Quantidade':'Qtde_Sala_Esterilização',

                        '2.4. Módulo de saúde: | Sala de esterilização | O espaço também é destinado a outras finalidades?':'Sala_Esterilização_Outros',

                        '2.4. Módulo de saúde: | Vestiário | O espaço está disponível no estabelecimento?':'Vestiario_Mod_Saude',

                        '2.4. Módulo de saúde: | Vestiário | Quantidade':'Qtde_Vestiario_Mod_Saude',

                        '2.4. Módulo de saúde: | Vestiário | O espaço também é destinado a outras finalidades?':'Vestiario_Mod_Saude_Outros',

                        '2.4. Módulo de saúde: | Depósito de material de limpeza - DML | O espaço está disponível no estabelecimento?':'Mod_Saude_DML',

                        '2.4. Módulo de saúde: | Depósito de material de limpeza - DML | Quantidade':'Qtde_Mod_Saude_DML',

                        '2.4. Módulo de saúde: | Depósito de material de limpeza - DML | O espaço também é destinado a outras finalidades?':'Mod_Saude_DML_Outros',

                        '2.4.1. Módulo de saúde: | Sala de atendimento clínico multiprofissional | O espaço está disponível no estabelecimento?':'Sala_Atend_Mult',

                        '2.4.1. Módulo de saúde: | Sala de atendimento clínico multiprofissional | Quantidade':'Qtde_Sala_Atend_Mult',

                        '2.4.1. Módulo de saúde: | Sala de atendimento clínico multiprofissional | O espaço também é destinado a outras finalidades?':'Sala_Atend_Mult_Outros',

                        '2.4.1. Módulo de saúde: | Sala de procedimentos | O espaço está disponível no estabelecimento?':'Sala_Proced',

                        '2.4.1. Módulo de saúde: | Sala de procedimentos | Quantidade':'Qtde_Sala_Proced',

                        '2.4.1. Módulo de saúde: | Sala de procedimentos | O espaço também é destinado a outras finalidades?':'Sala_Proced_Outros',

                        '2.4.1. Módulo de saúde: | Sala de raio x | O espaço está disponível no estabelecimento?': 'Sala_Raxio_X',

                        '2.4.1. Módulo de saúde: | Sala de raio x | Quantidade':'Qtde_Sala_Raxio_X',

                        '2.4.1. Módulo de saúde: | Sala de raio x | O espaço também é destinado a outras finalidades?':'Sala_Raxio_X_Outros',

                        '2.4.1. Módulo de saúde: | Laboratório de diagnóstico | O espaço está disponível no estabelecimento?':'Lab_Diagnostico',

                        '2.4.1. Módulo de saúde: | Laboratório de diagnóstico | Quantidade': 'Qtde_Lab_Diagnostico',

                        '2.4.1. Módulo de saúde: | Laboratório de diagnóstico | O espaço também é destinado a outras finalidades?':'Lab_Diagnostico_Outros',

                        '2.4.1. Módulo de saúde: | Cela de espera | O espaço está disponível no estabelecimento?':'Cela_Espera',

                        '2.4.1. Módulo de saúde: | Cela de espera | Quantidade':'Qtde_Cela_Espera',

                        '2.4.1. Módulo de saúde: | Cela de espera | O espaço também é destinado a outras finalidades?':'Cela_Espera_Outros',

                        '2.4.1. Módulo de saúde: | Solário para pacientes | O espaço está disponível no estabelecimento?':'Solario_Pacientes',

                        '2.4.1. Módulo de saúde: | Solário para pacientes | Quantidade':'Qtde_Solario_Pacientes',

                        '2.4.1. Módulo de saúde: | Solário para pacientes | O espaço também é destinado a outras finalidades?':'Solario_Pacientes_Outros',

                        '2.4.1. Módulo de saúde: | Outro(s). Qual(is)? (Especifique abaixo) | O espaço está disponível no estabelecimento?':'Espaco_Disponivel',

                        '2.4.1. Módulo de saúde: | Outro(s). Qual(is)? (Especifique abaixo) | Quantidade':'Qtde_Espaco_Disponivel',

                        '2.4.1. Módulo de saúde: | Outro(s). Qual(is)? (Especifique abaixo) | O espaço também é destinado a outras finalidades?':'Espaco_Disponivel_Outros',

                        '2.4.a. Módulo de saúde (Outros): | Outro 1 | Nome do espaço':'Nome_Espaco_Outro1',

                        '2.4.a. Módulo de saúde (Outros): | Outro 1 | Quantidade':'Qtde_Outro1',

                        '2.4.a. Módulo de saúde (Outros): | Outro 1 | O espaço também é destinado a outras finalidades?':'Desc_Outro1',

                        '2.4.a. Módulo de saúde (Outros): | Outro 2 | Nome do espaço':'Nome_Espaco_Outro2',

                        '2.4.a. Módulo de saúde (Outros): | Outro 2 | Quantidade':'Qtde_Outro2',

                        '2.4.a. Módulo de saúde (Outros): | Outro 2 | O espaço também é destinado a outras finalidades?':'Desc_Outro2',

                        '2.4.a. Módulo de saúde (Outros): | Outro 3 | Nome do espaço':'Nome_Espaco_Outro3',

                        '2.4.a. Módulo de saúde (Outros): | Outro 3 | Quantidade':'Qtde_Outro3',

                        '2.4.a. Módulo de saúde (Outros): | Outro 3 | O espaço também é destinado a outras finalidades?':'Desc_Outro3',

                        '2.4.a. Módulo de saúde (Outros): | Outro 4 | Nome do espaço':'Nome_Espaco_Outro4',

                        '2.4.a. Módulo de saúde (Outros): | Outro 4 | Quantidade':'Qtde_Outro4',

                        '2.4.a. Módulo de saúde (Outros): | Outro 4 | O espaço também é destinado a outras finalidades?':'Desc_Outro4',

                        '2.4.a. Módulo de saúde (Outros): | Outro 5 | Nome do espaço':'Nome_Espaco_Outro5',

                        '2.4.a. Módulo de saúde (Outros): | Outro 5 | Quantidade':'Qtde_Outro5',

                        '2.4.a. Módulo de saúde (Outros): | Outro 5 | O espaço também é destinado a outras finalidades?':'Desc_Outro5',

                        '2.4.a. Módulo de saúde (Outros): | Outro 6 | Nome do espaço':'Nome_Espaco_Outro6',

                        '2.4.a. Módulo de saúde (Outros): | Outro 6 | Quantidade':'Qtde_Outro6',

                        '2.4.a. Módulo de saúde (Outros): | Outro 6 | O espaço também é destinado a outras finalidades?':'Desc_Outro6',

                        '2.4.a. Módulo de saúde (Outros): | Outro 7 | Nome do espaço':'Nome_Espaco_Outro7',

                        '2.4.a. Módulo de saúde (Outros): | Outro 7 | Quantidade':'Qtde_Outro7',

                        '2.4.a. Módulo de saúde (Outros): | Outro 7 | O espaço também é destinado a outras finalidades?':'Desc_Outro7',

                        '2.4.a. Módulo de saúde (Outros): | Outro 8 | Nome do espaço':'Nome_Espaco_Outro8',

                        '2.4.a. Módulo de saúde (Outros): | Outro 8 | Quantidade':'Qtde_Outro8',

                        '2.4.a. Módulo de saúde (Outros): | Outro 8 | O espaço também é destinado a outras finalidades?':'Desc_Outro8',

                        '2.4.a. Módulo de saúde (Outros): | Outro 9 | Nome do espaço':'Nome_Espaco_Outro9',

                        '2.4.a. Módulo de saúde (Outros): | Outro 9 | Quantidade':'Qtde_Outro9',

                        '2.4.a. Módulo de saúde (Outros): | Outro 9 | O espaço também é destinado a outras finalidades?':'Desc_Outro9',

                        '2.4.a. Módulo de saúde (Outros): | Outro 10 | Nome do espaço':'Nome_Espaco_Outro10',

                        '2.4.a. Módulo de saúde (Outros): | Outro 10 | Quantidade':'Qtde_Outro10',

                        '2.4.a. Módulo de saúde (Outros): | Outro 10 | O espaço também é destinado a outras finalidades?':'Desc_Outro10',

                        '2.5. Módulo de educação: | Sala de aula | O espaço está disponível no estabelecimento?': 'Sala_Aula', 

                        '2.5. Módulo de educação: | Sala de aula | Quantidade de salas': 'Qtde_Salas_Aula',

                        '2.5. Módulo de educação: | Sala de aula | Capacidade para quantas pessoas?': 'Capacidade_Sala_Aula',

                        '2.5. Módulo de educação: | Sala de informática | O espaço está disponível no estabelecimento?': 'Sala_Inform',

                        '2.5. Módulo de educação: | Sala de informática | Quantidade de salas': 'Qtde_Salas_Inform',

                        '2.5. Módulo de educação: | Sala de informática | Capacidade para quantas pessoas?': 'Capacidade_Sala_Inform', 

                        '2.5. Módulo de educação: | Sala de encontros com a sociedade / sala de reuniões | O espaço está disponível no estabelecimento?': 'Sala_Encontro',

                        '2.5. Módulo de educação: | Sala de encontros com a sociedade / sala de reuniões | Quantidade de salas': 'Qtde_Sala_Encontro',

                        '2.5. Módulo de educação: | Sala de encontros com a sociedade / sala de reuniões | Capacidade para quantas pessoas?': 'Capacidade_Sala_Encontro',

                        '2.5. Módulo de educação: | Biblioteca | O espaço está disponível no estabelecimento?': 'Biblioteca',

                        '2.5. Módulo de educação: | Biblioteca | Quantidade de salas':'Qtde_Biblioteca',

                        '2.5. Módulo de educação: | Biblioteca | Capacidade para quantas pessoas?':'Biblioteca_Capacidade',

                        '2.5. Módulo de educação: | Sala de professores | O espaço está disponível no estabelecimento?': 'Sala_Professores',

                        '2.5. Módulo de educação: | Sala de professores | Quantidade de salas': 'Qtde_Sala_Professores',

                        '2.5. Módulo de educação: | Sala de professores | Capacidade para quantas pessoas?':'Sala_Professores_Capacidade',

                        '2.5. Módulo de educação: | Outro(s). Qual(is)? (Especifique abaixo) | O espaço está disponível no estabelecimento?':'Mod_Educacao_Outros',

                        '2.5. Módulo de educação: | Outro(s). Qual(is)? (Especifique abaixo) | Quantidade de salas':'Qtde_Mod_Educacao_Outros',

                        '2.5. Módulo de educação: | Outro(s). Qual(is)? (Especifique abaixo) | Capacidade para quantas pessoas?':'Mod_Educacao_Outros_Capacidade',

                        '2.5.a. Módulo de educação (Outros): | Outro 1 | Nome do espaço':'Mod_educacao_Outro1',

                        '2.5.a. Módulo de educação (Outros): | Outro 1 | Quantidade de salas':'Qtde_Mod_educacao_Outro1',

                        '2.5.a. Módulo de educação (Outros): | Outro 1 | Capacidade para quantas pessoas?':'Mod_educacao_Capacidade_Outro1',

                        '2.5.a. Módulo de educação (Outros): | Outro 2 | Nome do espaço':'Mod_educacao_Outro2',

                        '2.5.a. Módulo de educação (Outros): | Outro 2 | Quantidade de salas':'Qtde_Mod_educacao_Outro2',                         

                        '2.5.a. Módulo de educação (Outros): | Outro 2 | Capacidade para quantas pessoas?':'Mod_educacao_Capacidade_Outro2',

                        '2.5.a. Módulo de educação (Outros): | Outro 3 | Nome do espaço':'Mod_educacao_Outro3',

                        '2.5.a. Módulo de educação (Outros): | Outro 3 | Quantidade de salas':'Qtde_Mod_educacao_Outro3',

                        '2.5.a. Módulo de educação (Outros): | Outro 3 | Capacidade para quantas pessoas?':'Mod_educacao_Capacidade_Outro3',

                        '2.5.a. Módulo de educação (Outros): | Outro 4 | Nome do espaço':'Mod_educacao_Outro4',

                        '2.5.a. Módulo de educação (Outros): | Outro 4 | Quantidade de salas':'Qtde_Mod_educacao_Outro4',

                        '2.5.a. Módulo de educação (Outros): | Outro 4 | Capacidade para quantas pessoas?':'Mod_educacao_Capacidade_Outro4',

                        '2.5.a. Módulo de educação (Outros): | Outro 5 | Nome do espaço':'Mod_educacao_Outro5',

                        '2.5.a. Módulo de educação (Outros): | Outro 5 | Quantidade de salas':'Qtde_Mod_educacao_Outro5',

                        '2.5.a. Módulo de educação (Outros): | Outro 5 | Capacidade para quantas pessoas?':'Mod_educacao_Capacidade_Outro5',

                        '2.5.a. Módulo de educação (Outros): | Outro 6 | Nome do espaço':'Mod_educacao_Outro6',

                        '2.5.a. Módulo de educação (Outros): | Outro 6 | Quantidade de salas':'Qtde_Mod_educacao_Outro6',

                        '2.5.a. Módulo de educação (Outros): | Outro 6 | Capacidade para quantas pessoas?':'Mod_educacao_Capacidade_Outro6',

                        '2.5.a. Módulo de educação (Outros): | Outro 7 | Nome do espaço':'Mod_educacao_Outro7',

                        '2.5.a. Módulo de educação (Outros): | Outro 7 | Quantidade de salas':'Qtde_Mod_educacao_Outro7',

                        '2.5.a. Módulo de educação (Outros): | Outro 7 | Capacidade para quantas pessoas?':'Mod_educacao_Capacidade_Outro7',

                        '2.5.a. Módulo de educação (Outros): | Outro 8 | Nome do espaço':'Mod_educacao_Outro8',

                        '2.5.a. Módulo de educação (Outros): | Outro 8 | Quantidade de salas':'Qtde_Mod_educacao_Outro8',

                        '2.5.a. Módulo de educação (Outros): | Outro 8 | Capacidade para quantas pessoas?':'Mod_educacao_Capacidade_Outro8',

                        '2.5.a. Módulo de educação (Outros): | Outro 9 | Nome do espaço':'Mod_educacao_Outro9',

                        '2.5.a. Módulo de educação (Outros): | Outro 9 | Quantidade de salas':'Qtde_Mod_educacao_Outro9',

                        '2.5.a. Módulo de educação (Outros): | Outro 9 | Capacidade para quantas pessoas?':'Mod_educacao_Capacidade_Outro9',

                        '2.5.a. Módulo de educação (Outros): | Outro 10 | Nome do espaço':'Mod_educacao_Outro10',

                        '2.5.a. Módulo de educação (Outros): | Outro 10 | Quantidade de salas':'Qtde_Mod_educacao_Outro10',

                        '2.5.a. Módulo de educação (Outros): | Outro 10 | Capacidade para quantas pessoas?':'Mod_educacao_Capacidade_Outro10',

                        '2.6. Módulo de oficinas: [Não possui]': 'Oficina',

                        '2.6. Módulo de oficinas: [Sala de produção]':'Oficina_Producao',

                        '2.6. Módulo de oficinas: [Sala de controle/ supervisão]':'Oficina_Sala_Controle',

                        '2.6. Módulo de oficinas: [Sanitários]':'Oficina_Sanitarios',

                        '2.6. Módulo de oficinas: [Estoque]':'Oficina_Estoque',

                        '2.6. Módulo de oficinas: [Carga/ descarga]':'Oficina_Carga_Descarga',

                        '2.6. Módulo de oficinas: [Outro(s). Qual(is)?]':'Oficina_Outros',

                        '2.6. Módulo de oficinas: [Outro(s). Qual(is)?] [text]':'Oficina_Desc',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Artefatos de concreto | O módulo está disponível no estabelecimento?':'Oficina_Concreto',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Artefatos de concreto | Capacidade para quantas pessoas?':'Oficina_Concreto_Capacidade',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Blocos e tijolos | O módulo está disponível no estabelecimento?':'Oficina_Blocos',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Blocos e tijolos | Capacidade para quantas pessoas?':'Oficina_Blocos_Capacidade',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Padaria e panificação | O módulo está disponível no estabelecimento?':'Oficina_Padaria',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Padaria e panificação | Capacidade para quantas pessoas?':'Oficina_Padaria_Capacidade',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Corte e costura industrial | O módulo está disponível no estabelecimento?':'Oficina_Corte_Costura',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Corte e costura industrial | Capacidade para quantas pessoas?':'Oficina_Corte_Costura_Capacidade',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Artesanato | O módulo está disponível no estabelecimento?':'Oficina_Artesanato',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Artesanato | Capacidade para quantas pessoas?':'Oficina_Artesanato_Capacidade',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Marcenaria | O módulo está disponível no estabelecimento?':'Oficina_Marcenaria',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Marcenaria | Capacidade para quantas pessoas?':'Oficina_Marcenaria_Capacidade',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Serralheria | O módulo está disponível no estabelecimento?':'Oficina_Serralheria',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Serralheria | Capacidade para quantas pessoas?':'Oficina_Serralheria_Capacidade',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Outro(s). Qual(is)? (Especifique abaixo) | O módulo está disponível no estabelecimento?':'Oficina_Outros',

                        '2.6.1. Qual(is) módulo de oficina existe(m) no estabelecimento? | Outro(s). Qual(is)? (Especifique abaixo) | Capacidade para quantas pessoas?':'Oficina_Outros_Capacidade',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 1 | Nome da oficina':'Oficina_Outros1',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 1 | Capacidade para quantas pessoas?':'Oficina_Capacidade_Outros1',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 2 | Nome da oficina':'Oficina_Outros2',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 2 | Capacidade para quantas pessoas?':'Oficina_Capacidade_Outros2',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 3 | Nome da oficina':'Oficina_Outros3',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 3 | Capacidade para quantas pessoas?':'Oficina_Capacidade_Outros3',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 4 | Nome da oficina':'Oficina_Outros4',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 4 | Capacidade para quantas pessoas?':'Oficina_Capacidade_Outros4',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 5 | Nome da oficina':'Oficina_Outros5',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 5 | Capacidade para quantas pessoas?':'Oficina_Capacidade_Outros5',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 6 | Nome da oficina':'Oficina_Outros6',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 6 | Capacidade para quantas pessoas?':'Oficina_Capacidade_Outros6',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 7 | Nome da oficina':'Oficina_Capacidade_Outros7',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 7 | Capacidade para quantas pessoas?':'Oficina_Capacidade_Outros7',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 8 | Nome da oficina':'Oficina_Outros8',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 8 | Capacidade para quantas pessoas?':'Oficina_Capacidade_Outros8',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 9 | Nome da oficina':'Oficina_Outros9',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 9 | Capacidade para quantas pessoas?':'Oficina_Capacidade_Outros9',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 10 | Nome da oficina':'Oficina_Outros10',

                        '2.6.a. Módulo de oficinas (Outros): | Outro 10 | Capacidade para quantas pessoas?':'Oficina_Capacidade_Outros10',

                        '2.7. Há local específico para visitação?':'Local_Visitacao',

                        '2.8. Há local específico para visita íntima?':'Local_Visita_Intima',

                        '2.9. Há sala de atendimento para serviço social?':'Sala_Serv_Social',

                        '2.10. Há sala de atendimento para psicologia?':'Sala_Psicologia',

                        '2.11. Há local destinado ao atendimento jurídico gratuito no estabelecimento?':'Sala_Juridico_Grat',

                        '2.12. Possui sala de videoconferência?':'Sala_Video_Conf',

                        '2.13. Há "cela(s)-seguro"?':'Cela_Seguro',

                        '2.14. Há ala ou cela destinadas exclusivamente às pessoas privadas de liberdade que se declarem lésbicas, gays, bissexuais, travestis e transexuais (LGBT)?':'Cela_LGBT',

                        '2.14. Há ala ou cela destinadas exclusivamente às pessoas privadas de liberdade que se declarem lésbicas, gays, bissexuais, travestis e transexuais (LGBT)? [other]':'Cela_LGBT_Outros',

                        '2.15. Há ala ou cela destinada exclusivamente para idosos?':'Cela_Idosos',

                        '2.15. Há ala ou cela destinada exclusivamente para idosos? [other]':'Cela_Idosos_Outros',

                        '2.16. Há ala ou cela destinada exclusivamente para indígenas?':'Cela_Indigena',

                        '2.16. Há ala ou cela destinada exclusivamente para indígenas? [other]':'Cela_Indigena_Outro',

                        '2.17. Há ala ou cela destinada exclusivamente para pessoas estrangeiras?':'Cela_Estrangeiros',

                        '2.17. Há ala ou cela destinada exclusivamente para pessoas estrangeiras? [other]':'Cela_Estrangeiros_Outros',

                        '2.18. Há acessibilidade para pessoas com deficiência?':'Acessibilidade',

                        '2.18. Há acessibilidade para pessoas com deficiência? [other]':'Acessibilidade_Outro',

                        '2.19. Há terreno/ espaço disponível para construção de novos módulos?':'Espaco_Disponível',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Cargos administrativos (atribuição de cunho estritamente administrativo) | Efetivo Masculino':'Qtde_Cargo_Adm_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Cargos administrativos (atribuição de cunho estritamente administrativo) | Efetivo Feminino':'Qtde_Cargo_Adm_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Cargos administrativos (atribuição de cunho estritamente administrativo) | Comissionado Masculino':'Qtde_Cargo_Comissionado_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Cargos administrativos (atribuição de cunho estritamente administrativo) | Comissionado Feminino':'Qtde_Cargo_Comissionado_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Cargos administrativos (atribuição de cunho estritamente administrativo) | Terceirizado Masculino':'Qtde_Cargo_Adm_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Cargos administrativos (atribuição de cunho estritamente administrativo) | Terceirizado Feminino':'Qtde_Cargo_Adm_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Cargos administrativos (atribuição de cunho estritamente administrativo) | Temporário Masculino':'Qtde_Cargo_Adm_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Cargos administrativos (atribuição de cunho estritamente administrativo) | Temporário Feminino':'Qtde_Cargo_Adm_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Cargos administrativos (atribuição de cunho estritamente administrativo) | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Cargo_Adm_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Enfermeiros | Efetivo Masculino':'Qtde_Enfermeiros',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Enfermeiros | Efetivo Feminino':'Qtde_Enfermeiras',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Enfermeiros | Comissionado Masculino':'Enfermeiros_Comissionado',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Enfermeiros | Comissionado Feminino':'Qtde_Enfermeiras_Comissionadas',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Enfermeiros | Terceirizado Masculino':'Qtde_Enfermeiros_Terc',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Enfermeiros | Terceirizado Feminino':'Qtde_Enfermeiras_Terc',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Enfermeiros | Temporário Masculino':'Qtde_Enfermeiros_Temp',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Enfermeiros | Temporário Feminino':'Qtde_Enfermeiras_Temp',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Enfermeiros | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Enfermeiros_Adm_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Auxiliar e técnico de enfermagem | Efetivo Masculino':'Qtde_Tec_Enfermagem_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Auxiliar e técnico de enfermagem | Efetivo Feminino':'Qtde_Tec_Enfermagem_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Auxiliar e técnico de enfermagem | Comissionado Masculino':'Qtde_Tec_Enfermagem_Comissionado_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Auxiliar e técnico de enfermagem | Comissionado Feminino':'Qtde_Tec_Enfermagem_Comissionado_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Auxiliar e técnico de enfermagem | Terceirizado Masculino':'Qtde_Tec_Enfermagem_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Auxiliar e técnico de enfermagem | Terceirizado Feminino':'Qtde_Tec_Enfermagem_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Auxiliar e técnico de enfermagem | Temporário Masculino':'Qtde_Tec_Enfermagem_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Auxiliar e técnico de enfermagem | Temporário Feminino':'Qtde_Tec_Enfermagem_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Auxiliar e técnico de enfermagem | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Enfermagem_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Psicólogos | Efetivo Masculino':'Qtde_Psicologos',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Psicólogos | Efetivo Feminino':'Qtde_Psicologas',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Psicólogos | Comissionado Masculino':'Qtde_Psicologos_Comissionados',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Psicólogos | Comissionado Feminino':'Qtde_Psicologas_Comissionadas',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Psicólogos | Terceirizado Masculino':'Qtde_Psicologos_Terc',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Psicólogos | Terceirizado Feminino':'Qtde_Psicologas_Terc',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Psicólogos | Temporário Masculino':'Qtde_Psicologos_Temp',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Psicólogos | Temporário Feminino':'Qtde_Psicologas_Temp',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Psicólogos | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Psicologos_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Dentistas | Efetivo Masculino':'Qtde_Dentista_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Dentistas | Efetivo Feminino':'Qtde_Dentista_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Dentistas | Comissionado Masculino':'Qtde_Dentista_Comissionado_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Dentistas | Comissionado Feminino':'Qtde_Dentista_Comissionado_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Dentistas | Terceirizado Masculino':'Qtde_Denstista_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Dentistas | Terceirizado Feminino':'Qtde_Denstista_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Dentistas | Temporário Masculino':'Qtde_Dentista_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Dentistas | Temporário Feminino':'Qtde_Dentista_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Dentistas | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Dentistas_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Técnico/ auxiliar odontológico | Efetivo Masculino':'Qtde_Aux_Odont_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Técnico/ auxiliar odontológico | Efetivo Feminino':'Qtde_Aux_Odont_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Técnico/ auxiliar odontológico | Comissionado Masculino':'Qtde_Aux_Odont_Comissionado_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Técnico/ auxiliar odontológico | Comissionado Feminino':'Qtde_Aux_Odont_Comissionado_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Técnico/ auxiliar odontológico | Terceirizado Masculino':'Qtde_Aux_Odont_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Técnico/ auxiliar odontológico | Terceirizado Feminino':'Qtde_Aux_Odont_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Técnico/ auxiliar odontológico | Temporário Masculino':'Qtde_Aux_Odont_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Técnico/ auxiliar odontológico | Temporário Feminino':'Qtde_Aux_Odont_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Técnico/ auxiliar odontológico | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Aux_Odont_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Assistentes sociais | Efetivo Masculino':'Qtde_Assist_Social_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Assistentes sociais | Efetivo Feminino':'Qtde_Assist_Social_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Assistentes sociais | Comissionado Masculino':'Qtde_Assist_Social_Comissionado_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Assistentes sociais | Comissionado Feminino':'Qtde_Assist_Social_Comissionado_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Assistentes sociais | Terceirizado Masculino':'Qtde_Assist_Social_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Assistentes sociais | Terceirizado Feminino':'Qtde_Assist_Social_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Assistentes sociais | Temporário Masculino':'Qtde_Assist_Social_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Assistentes sociais | Temporário Feminino':'Qtde_Assist_Social_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Assistentes sociais | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Assist_Social_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Advogados | Efetivo Masculino':'Qtde_Advogados',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Advogados | Efetivo Feminino':'Qtde_Advogadas',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Advogados | Comissionado Masculino':'Qtde_Advogados_Comissionados',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Advogados | Comissionado Feminino':'Qtde_Advogadas_Comissionadas',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Advogados | Terceirizado Masculino':'Qtde_Advogados_Terc',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Advogados | Terceirizado Feminino':'Qtde_Advogadas_Terc',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Advogados | Temporário Masculino':'Qtde_Advogados_Temp',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Advogados | Temporário Feminino':'Qtde_Advogadas_Temp',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Advogados | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Advogados_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - clínicos gerais | Efetivo Masculino':'Qtde_Medicos',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - clínicos gerais | Efetivo Feminino':'Qtde_Medicas',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - clínicos gerais | Comissionado Masculino':'Qtde_Medicos_Comissionados',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - clínicos gerais | Comissionado Feminino':'Qtde_Medicas_Comissionados',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - clínicos gerais | Terceirizado Masculino':'Qtde_Medicos_Terc',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - clínicos gerais | Terceirizado Feminino':'Qtde_Medicas_Terc',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - clínicos gerais | Temporário Masculino':'Qtde_Medicos_Temp',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - clínicos gerais | Temporário Feminino':'Qtde_Medicas_Temp',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - clínicos gerais | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Medicos_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - ginecologistas | Efetivo Masculino':'Qtde_Ginecologistas_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - ginecologistas | Efetivo Feminino':'Qtde_Ginecologistas_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - ginecologistas | Comissionado Masculino':'Qtde_Ginecologistas_Comissionados_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - ginecologistas | Comissionado Feminino':'Qtde_Ginecologistas_Comissionados_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - ginecologistas | Terceirizado Masculino':'Qtde_Ginecologistas_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - ginecologistas | Terceirizado Feminino':'Qtde_Ginecologistas_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - ginecologistas | Temporário Masculino':'Qtde_Ginecologitas_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - ginecologistas | Temporário Feminino':'Qtde_Ginecologitas_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - ginecologistas | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Ginecologostas_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - psiquiatras | Efetivo Masculino':'Qtde_Psiquiatras_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - psiquiatras | Efetivo Feminino':'Qtde_Psiquiatras_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - psiquiatras | Comissionado Masculino':'Qtde_Psiquiatras_Comissionados_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - psiquiatras | Comissionado Feminino':'Qtde_Psiquiatras_Comissionados_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - psiquiatras | Terceirizado Masculino':'Qtde_Psiquiatras_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - psiquiatras | Terceirizado Feminino':'Qtde_Psiquiatras_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - psiquiatras | Temporário Masculino':'Qtde_Psiquiatras_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - psiquiatras | Temporário Feminino':'Qtde_Psiquiatras_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - psiquiatras | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Psiquiatras_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - outras especialidades | Efetivo Masculino':'Qtde_Medicos_Outros_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - outras especialidades | Efetivo Feminino':'Qtde_Medicos_Outros_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - outras especialidades | Comissionado Masculino':'Qtde_Medicos_Outros_Comissionados_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - outras especialidades | Comissionado Feminino':'Qtde_Medicos_Outros_Comissionados_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - outras especialidades | Terceirizado Masculino':'Qtde_Medicos_Outros_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - outras especialidades | Terceirizado Feminino':'Qtde_Medicos_Outros_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - outras especialidades | Temporário Masculino':'Qtde_Medicos_Outros_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - outras especialidades | Temporário Feminino':'Qtde_Medicos_Outros_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Médicos - outras especialidades | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Medicos_Outros_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Pedagogos | Efetivo Masculino':'Qtde_Pedagogos_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Pedagogos | Efetivo Feminino':'Qtde_Pedagogos_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Pedagogos | Comissionado Masculino':'Qtde_Pedagogos_Comissionados_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Pedagogos | Comissionado Feminino':'Qtde_Pedagogos_Comissionados_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Pedagogos | Terceirizado Masculino':'Qtde_Pedagogos_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Pedagogos | Terceirizado Feminino':'Qtde_Pedagogos_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Pedagogos | Temporário Masculino':'Qtde_Pedagogos_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Pedagogos | Temporário Feminino':'Qtde_Pedagogos_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Pedagogos | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Pedagogos_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Professores | Efetivo Masculino':'Qtde_Professores_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Professores | Efetivo Feminino':'Qtde_Professores_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Professores | Comissionado Masculino':'Qtde_Professores_Comissionados_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Professores | Comissionado Feminino':'Qtde_Professores_Comissionados_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Professores | Terceirizado Masculino':'Qtde_Professores_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Professores | Terceirizado Feminino':'Qtde_Professores_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Professores | Temporário Masculino':'Qtde_Professores_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Professores | Temporário Feminino':'Qtde_Professores_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Professores | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Professores_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Terapeuta/ terapeuta ocupacional | Efetivo Masculino':'Qtde_Terapeutas_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Terapeuta/ terapeuta ocupacional | Efetivo Feminino':'Qtde_Terapeutas_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Terapeuta/ terapeuta ocupacional | Comissionado Masculino':'Qtde_Terapeutas_Comissionados_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Terapeuta/ terapeuta ocupacional | Comissionado Feminino':'Qtde_Terapeutas_Comissionados_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Terapeuta/ terapeuta ocupacional | Terceirizado Masculino':'Qtde_Terapeutas_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Terapeuta/ terapeuta ocupacional | Terceirizado Feminino':'Qtde_Terapeutas_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Terapeuta/ terapeuta ocupacional | Temporário Masculino':'Qtde_Terapeutas_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Terapeuta/ terapeuta ocupacional | Temporário Feminino':'Qtde_Terapeutas_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Terapeuta/ terapeuta ocupacional | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Terapeutas_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Civil em atividade exclusiva no estabelecimento prisional | Efetivo Masculino':'Qtde_PC_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Civil em atividade exclusiva no estabelecimento prisional | Efetivo Feminino':'Qtde_PC_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Civil em atividade exclusiva no estabelecimento prisional | Comissionado Masculino':'Qtde_PC_Comissionado_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Civil em atividade exclusiva no estabelecimento prisional | Comissionado Feminino':'Qtde_PC_Comissionado_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Civil em atividade exclusiva no estabelecimento prisional | Terceirizado Masculino':'Qtde_PC_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Civil em atividade exclusiva no estabelecimento prisional | Terceirizado Feminino':'Qtde_PC_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Civil em atividade exclusiva no estabelecimento prisional | Temporário Masculino':'Qtde_PC_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Civil em atividade exclusiva no estabelecimento prisional | Temporário Feminino':'Qtde_PC_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Civil em atividade exclusiva no estabelecimento prisional | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_PC_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Militar em atividade exclusiva no estabelecimento prisional | Efetivo Masculino':'Qtde_PM_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Militar em atividade exclusiva no estabelecimento prisional | Efetivo Feminino':'Qtde_PM_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Militar em atividade exclusiva no estabelecimento prisional | Comissionado Masculino':'Qtde_PM_Comissionado_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Militar em atividade exclusiva no estabelecimento prisional | Comissionado Feminino':'Qtde_PM_Comissionado_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Militar em atividade exclusiva no estabelecimento prisional | Terceirizado Masculino':'Qtde_PM_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Militar em atividade exclusiva no estabelecimento prisional | Terceirizado Feminino':'Qtde_PM_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Militar em atividade exclusiva no estabelecimento prisional | Temporário Masculino':'Qtde_PM_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Militar em atividade exclusiva no estabelecimento prisional | Temporário Feminino':'Qtde_PM_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Policial Militar em atividade exclusiva no estabelecimento prisional | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_PM_Lot_Orig',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Outros. (Especificar abaixo) | Efetivo Masculino':'Qtde_Servidores_Outros_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Outros. (Especificar abaixo) | Efetivo Feminino':'Qtde_Servidores_Outros_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Outros. (Especificar abaixo) | Comissionado Masculino':'Qtde_Servidores_Outros_Comissionados_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Outros. (Especificar abaixo) | Comissionado Feminino':'Qtde_Servidores_Outros_Comissionados_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Outros. (Especificar abaixo) | Terceirizado Masculino':'Qtde_Servidores_Outros_Terc_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Outros. (Especificar abaixo) | Terceirizado Feminino':'Qtde_Servidores_Outros_Terc_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Outros. (Especificar abaixo) | Temporário Masculino':'Qtde_Servidores_Outros_Temp_M',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Outros. (Especificar abaixo) | Temporário Feminino':'Qtde_Servidores_Outros_Temp_F',

                        '3.1. Quantidade de Servidores que atuam no Sistema Prisional | Outros. (Especificar abaixo) | Órgão de lotação originária  (para efetivos e comissionados)':'Qtde_Servidores_Outros_Lot_Orig',

                        '3.1.a. Cargos/Funções (Outros): | Outro 1 | Nome do cargo/função':'Cargos_Outros1',

                        '3.1.a. Cargos/Funções (Outros): | Outro 1 | Efetivo Masculino':'Cargos_Outros1_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 1 | Efetivo Feminino':'Cargos_Outros1_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 1 | Comissionado Masculino':'Cargos_Outros1_Comissionado_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 1 | Comissionado Feminino':'Cargos_Outros1_Comissionado_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 1 | Terceirizado Masculino':'Cargos_Outros1_Terc_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 1 | Terceirizado Feminino':'Cargos_Outros1_Terc_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 1 | Temporário Masculino':'Cargos_Outros1_Temp_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 1 | Temporário Feminino':'Cargos_Outros1_Temp_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 1 | Órgão de lotação originária':'Cargos_Outros1_Lot_Orig',

                        '3.1.a. Cargos/Funções (Outros): | Outro 2 | Nome do cargo/função':'Cargos_Outros2',

                        '3.1.a. Cargos/Funções (Outros): | Outro 2 | Efetivo Masculino':'Cargos_Outros2_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 2 | Efetivo Feminino':'Cargos_Outros2_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 2 | Comissionado Masculino':'Cargos_Outros2_Comissionado_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 2 | Comissionado Feminino':'Cargos_Outros2_Comissionado_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 2 | Terceirizado Masculino':'Cargos_Outros2_Terc_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 2 | Terceirizado Feminino':'Cargos_Outros2_Terc_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 2 | Temporário Masculino':'Cargos_Outros2_Temp_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 2 | Temporário Feminino':'Cargos_Outros2_Temp_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 2 | Órgão de lotação originária':'Cargos_Outros2_Lot_Orig',

                        '3.1.a. Cargos/Funções (Outros): | Outro 3 | Nome do cargo/função':'Cargos_Outros3',

                        '3.1.a. Cargos/Funções (Outros): | Outro 3 | Efetivo Masculino':'Cargos_Outros3_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 3 | Efetivo Feminino':'Cargos_Outros3_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 3 | Comissionado Masculino':'Cargos_Outros3_Comissionado_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 3 | Comissionado Feminino':'Cargos_Outros3_Comissionado_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 3 | Terceirizado Masculino':'Cargos_Outros3_Terc_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 3 | Terceirizado Feminino':'Cargos_Outros3_Terc_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 3 | Temporário Masculino':'Cargos_Outros3_Temp_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 3 | Temporário Feminino':'Cargos_Outros3_Temp_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 3 | Órgão de lotação originária':'Cargos_Outros3_Lot_Orig',

                        '3.1.a. Cargos/Funções (Outros): | Outro 4 | Nome do cargo/função':'Cargos_Outros4',

                        '3.1.a. Cargos/Funções (Outros): | Outro 4 | Efetivo Masculino':'Cargos_Outros4_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 4 | Efetivo Feminino':'Cargos_Outros4_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 4 | Comissionado Masculino':'Cargos_Outros4_Comissionado_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 4 | Comissionado Feminino':'Cargos_Outros4_Comissionado_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 4 | Terceirizado Masculino':'Cargos_Outros4_Terc_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 4 | Terceirizado Feminino':'Cargos_Outros4_Terc_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 4 | Temporário Masculino':'Cargos_Outros4_Temp_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 4 | Temporário Feminino':'Cargos_Outros4_Temp_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 4 | Órgão de lotação originária':'Cargos_Outros4_Lot_Orig',

                        '3.1.a. Cargos/Funções (Outros): | Outro 5 | Nome do cargo/função':'Cargos_Outros5',

                        '3.1.a. Cargos/Funções (Outros): | Outro 5 | Efetivo Masculino':'Cargos_Outros5_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 5 | Efetivo Feminino':'Cargos_Outros5_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 5 | Comissionado Masculino':'Cargos_Outros5_Comissionado_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 5 | Comissionado Feminino':'Cargos_Outros5_Comissionado_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 5 | Terceirizado Masculino':'Cargos_Outros5_Terc_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 5 | Terceirizado Feminino':'Cargos_Outros5_Terc_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 5 | Temporário Masculino':'Cargos_Outros5_Temp_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 5 | Temporário Feminino':'Cargos_Outros5_Temp_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 5 | Órgão de lotação originária':'Cargos_Outros5_Lot_Orig',

                        '3.1.a. Cargos/Funções (Outros): | Outro 6 | Nome do cargo/função':'Cargos_Outros6',

                        '3.1.a. Cargos/Funções (Outros): | Outro 6 | Efetivo Masculino':'Cargos_Outros6_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 6 | Efetivo Feminino':'Cargos_Outros6_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 6 | Comissionado Masculino':'Cargos_Outros6_Comissionado_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 6 | Comissionado Feminino':'Cargos_Outros6_Comissionado_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 6 | Terceirizado Masculino':'Cargos_Outros6_Terc_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 6 | Terceirizado Feminino':'Cargos_Outros6_Terc_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 6 | Temporário Masculino':'Cargos_Outros6_Temp_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 6 | Temporário Feminino':'Cargos_Outros6_Temp_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 6 | Órgão de lotação originária':'Cargos_Outros6_Lot_Orig',

                        '3.1.a. Cargos/Funções (Outros): | Outro 7 | Nome do cargo/função':'Cargos_Outros7',

                        '3.1.a. Cargos/Funções (Outros): | Outro 7 | Efetivo Masculino':'Cargos_Outros7_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 7 | Efetivo Feminino':'Cargos_Outros7_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 7 | Comissionado Masculino':'Cargos_Outros7_Comissionado_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 7 | Comissionado Feminino':'Cargos_Outros7_Comissionado_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 7 | Terceirizado Masculino':'Cargos_Outros7_Terc_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 7 | Terceirizado Feminino':'Cargos_Outros7_Terc_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 7 | Temporário Masculino':'Cargos_Outros7_Temp_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 7 | Temporário Feminino':'Cargos_Outros7_Temp_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 7 | Órgão de lotação originária':'Cargos_Outros7_Lot_Orig',

                        '3.1.a. Cargos/Funções (Outros): | Outro 8 | Nome do cargo/função':'Cargos_Outros8',

                        '3.1.a. Cargos/Funções (Outros): | Outro 8 | Efetivo Masculino':'Cargos_Outros8_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 8 | Efetivo Feminino':'Cargos_Outros8_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 8 | Comissionado Masculino':'Cargos_Outros8_Comissionado_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 8 | Comissionado Feminino':'Cargos_Outros8_Comissionado_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 8 | Terceirizado Masculino':'Cargos_Outros8_Terc_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 8 | Terceirizado Feminino':'Cargos_Outros8_Terc_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 8 | Temporário Masculino':'Cargos_Outros8_Temp_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 8 | Temporário Feminino':'Cargos_Outros8_Temp_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 8 | Órgão de lotação originária':'Cargos_Outros8_Lot_Orig',

                        '3.1.a. Cargos/Funções (Outros): | Outro 9 | Nome do cargo/função':'Cargos_Outros9',

                        '3.1.a. Cargos/Funções (Outros): | Outro 9 | Efetivo Masculino':'Cargos_Outros9_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 9 | Efetivo Feminino':'Cargos_Outros9_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 9 | Comissionado Masculino':'Cargos_Outros9_Comissionado_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 9 | Comissionado Feminino':'Cargos_Outros9_Comissionado_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 9 | Terceirizado Masculino':'Cargos_Outros9_Terc_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 9 | Terceirizado Feminino':'Cargos_Outros9_Terc_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 9 | Temporário Masculino':'Cargos_Outros9_Temp_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 9 | Temporário Feminino':'Cargos_Outros9_Temp_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 9 | Órgão de lotação originária':'Cargos_Outros9_Lot_Orig',

                        '3.1.a. Cargos/Funções (Outros): | Outro 10 | Nome do cargo/função':'Cargos_Outros10',

                        '3.1.a. Cargos/Funções (Outros): | Outro 10 | Efetivo Masculino':'Cargos_Outros10_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 10 | Efetivo Feminino':'Cargos_Outros10_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 10 | Comissionado Masculino':'Cargos_Outros10_Comissionado_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 10 | Comissionado Feminino':'Cargos_Outros10_Comissionado_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 10 | Terceirizado Masculino':'Cargos_Outros10_Terc_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 10 | Terceirizado Feminino':'Cargos_Outros10_Terc_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 10 | Temporário Masculino':'Cargos_Outros10_Temp_M',

                        '3.1.a. Cargos/Funções (Outros): | Outro 10 | Temporário Feminino':'Cargos_Outros10_Temp_F',

                        '3.1.a. Cargos/Funções (Outros): | Outro 10 | Órgão de lotação originária':'Cargos_Outros10_Lot_Orig',

                        '3.2. Há equipe própria para atendimento no berçário e/ou creche? [Sim, há médico pediatra]':'Pediatra',

                        '3.2. Há equipe própria para atendimento no berçário e/ou creche? [Sim, há médico ginecologista]':'Ginecologista',

                        '3.2. Há equipe própria para atendimento no berçário e/ou creche? [Sim, há nutricionista]':'Nutricionista',

                        '3.2. Há equipe própria para atendimento no berçário e/ou creche? [Sim, há cuidadores/as]':'Cuidadores',

                        '3.2. Há equipe própria para atendimento no berçário e/ou creche? [Sim, outro(s). Especificar:]':'Outros',

                        '3.2. Há equipe própria para atendimento no berçário e/ou creche? [Sim, outro(s). Especificar:] [text]':'Outros_Desc',

                        '3.2. Há equipe própria para atendimento no berçário e/ou creche? [Não, os atendimentos são realizados externamente]':'Equipe_Propria_Creche',

                        '3.2. Há equipe própria para atendimento no berçário e/ou creche? [Não se aplica (estabelecimento masculino)]':'Nao_Equipe_Propria_Creche',

                        '3.3. Há prestação sistemática de assistência jurídica gratuita às pessoas privadas de liberdade neste estabelecimento? [Não]':'Assist_Juridica',

                        '3.3. Há prestação sistemática de assistência jurídica gratuita às pessoas privadas de liberdade neste estabelecimento? [Sim, por meio da Defensoria Pública]':'Assist_Juridica_Defensoria_Publ',

                        '3.3. Há prestação sistemática de assistência jurídica gratuita às pessoas privadas de liberdade neste estabelecimento? [Sim, por meio de assistência jurídica privada prestada por advogados conveniados/ dativos]':'Assist_Juridica_Defensoria_Priv',

                        '3.3. Há prestação sistemática de assistência jurídica gratuita às pessoas privadas de liberdade neste estabelecimento? [Sim, por meio de assistência jurídica privada prestada por ONG ou outra entidade sem fins lucrativos]':'Assist_Juridica_Defensoria_ONG',

                        '3.3. Há prestação sistemática de assistência jurídica gratuita às pessoas privadas de liberdade neste estabelecimento? [Sim, outro. Qual?]':'Assist_Juridica_Outro',

                        '3.3. Há prestação sistemática de assistência jurídica gratuita às pessoas privadas de liberdade neste estabelecimento? [Sim, outro. Qual?] [text]':'Assist_Juridica_Outro_Desc',

                        '[Q_4_1.0.0] 4.1. População prisional: | Presos provisórios (sem condenação)** | Justiça Estadual Masculino':'Pop_Prisional_Prov_Just_Est_M',

                        '[Q_4_1.1.0] 4.1. População prisional: | Presos provisórios (sem condenação)** | Justiça Estadual Feminino':'Pop_Prisional_Prov_Just_Est_F',

                        '[Q_4_1.2.0] 4.1. População prisional: | Presos provisórios (sem condenação)** | Justiça Federal Masculino':'Pop_Prisional_Prov_Just_Fed_M',

                        '[Q_4_1.3.0] 4.1. População prisional: | Presos provisórios (sem condenação)** | Justiça Federal Feminino':'Pop_Prisional_Prov_Just_Fed_F',

                        '[Q_4_1.4.0] 4.1. População prisional: | Presos provisórios (sem condenação)** | Outros  (Justiça do Trabalho, Cível) Masculino':'Pop_Prisional_Prov_Outros_M',

                        '[Q_4_1.5.0] 4.1. População prisional: | Presos provisórios (sem condenação)** | Outros  (Justiça do Trabalho, Cível) Feminino':'Pop_Prisional_Prov_Outros_F',

                        '[Q_4_1.0.1] 4.1. População prisional: | Presos sentenciados - regime fechado | Justiça Estadual Masculino':'Pop_Prisional_Fechado_Just_Est_M',

                        '[Q_4_1.1.1] 4.1. População prisional: | Presos sentenciados - regime fechado | Justiça Estadual Feminino':'Pop_Prisional_Fechado_Just_Est_F',

                        '[Q_4_1.2.1] 4.1. População prisional: | Presos sentenciados - regime fechado | Justiça Federal Masculino':'Pop_Prisional_Fechado_Just_Fed_M',

                        '[Q_4_1.3.1] 4.1. População prisional: | Presos sentenciados - regime fechado | Justiça Federal Feminino':'Pop_Prisional_Fechado_Just_Fed_F',

                        '[Q_4_1.4.1] 4.1. População prisional: | Presos sentenciados - regime fechado | Outros  (Justiça do Trabalho, Cível) Masculino':'Pop_Prisional_Fechado_Outros_M',

                        '[Q_4_1.5.1] 4.1. População prisional: | Presos sentenciados - regime fechado | Outros  (Justiça do Trabalho, Cível) Feminino':'Pop_Prisional_Fechado_Outros_F',

                        '[Q_4_1.0.2] 4.1. População prisional: | Presos sentenciados - regime semiaberto | Justiça Estadual Masculino':'Pop_Prisional_Semi_Just_Est_M',

                        '[Q_4_1.1.2] 4.1. População prisional: | Presos sentenciados - regime semiaberto | Justiça Estadual Feminino':'Pop_Prisional_Semi_Just_Est_F',

                        '[Q_4_1.2.2] 4.1. População prisional: | Presos sentenciados - regime semiaberto | Justiça Federal Masculino':'Pop_Prisional_Semi_Just_Fed_M',

                        '[Q_4_1.3.2] 4.1. População prisional: | Presos sentenciados - regime semiaberto | Justiça Federal Feminino':'Pop_Prisional_Semi_Just_Fed_F',

                        '[Q_4_1.4.2] 4.1. População prisional: | Presos sentenciados - regime semiaberto | Outros  (Justiça do Trabalho, Cível) Masculino':'Pop_Prisional_Semi_Outros_M',

                        '[Q_4_1.5.2] 4.1. População prisional: | Presos sentenciados - regime semiaberto | Outros  (Justiça do Trabalho, Cível) Feminino':'Pop_Prisional_Semi_Outros_F',

                        '[Q_4_1.0.3] 4.1. População prisional: | Presos sentenciados - regime aberto | Justiça Estadual Masculino':'Pop_Prisional_Aberto_Just_Est_M',

                        '[Q_4_1.1.3] 4.1. População prisional: | Presos sentenciados - regime aberto | Justiça Estadual Feminino':'Pop_Prisional_Aberto_Just_Est_F',

                        '[Q_4_1.2.3] 4.1. População prisional: | Presos sentenciados - regime aberto | Justiça Federal Masculino':'Pop_Prisional_Aberto_Just_Fed_F',

                        '[Q_4_1.3.3] 4.1. População prisional: | Presos sentenciados - regime aberto | Justiça Federal Feminino':'Pop_Prisional_Aberto_Just_Fed_M',

                        '[Q_4_1.4.3] 4.1. População prisional: | Presos sentenciados - regime aberto | Outros  (Justiça do Trabalho, Cível) Masculino':'Pop_Prisional_Aberto_Outros_M',

                        '[Q_4_1.5.3] 4.1. População prisional: | Presos sentenciados - regime aberto | Outros  (Justiça do Trabalho, Cível) Feminino':'Pop_Prisional_Aberto_Outros_F',

                        '[Q_4_1.0.4] 4.1. População prisional: | Medida de segurança - internação | Justiça Estadual Masculino':'Pop_Prisional_Med_Seg_Just_Est_Internacao_M',

                        '[Q_4_1.1.4] 4.1. População prisional: | Medida de segurança - internação | Justiça Estadual Feminino':'Pop_Prisional_Med_Seg_Just_Est_Internacao_F',

                        '[Q_4_1.2.4] 4.1. População prisional: | Medida de segurança - internação | Justiça Federal Masculino':'Pop_Prisional_Med_Seg_Just_Fed_Internacao_M',

                        '[Q_4_1.3.4] 4.1. População prisional: | Medida de segurança - internação | Justiça Federal Feminino':'Pop_Prisional_Med_Seg_Just_Fed_Internacao_F',

                        '[Q_4_1.4.4] 4.1. População prisional: | Medida de segurança - internação | Outros  (Justiça do Trabalho, Cível) Masculino':'Pop_Prisional_Med_Seg_Outros_Internacao_M',

                        '[Q_4_1.5.4] 4.1. População prisional: | Medida de segurança - internação | Outros  (Justiça do Trabalho, Cível) Feminino':'Pop_Prisional_Med_Seg_Outros_Internacao_F',

                        '[Q_4_1.0.5] 4.1. População prisional: | Medida de segurança - tratamento ambulatorial | Justiça Estadual Masculino':'Pop_Prisional_Med_Seg_Just_Est_Ambulatorial_M',

                        '[Q_4_1.1.5] 4.1. População prisional: | Medida de segurança - tratamento ambulatorial | Justiça Estadual Feminino':'Pop_Prisional_Med_Seg_Just_Est_Ambulatorial_F',

                        '[Q_4_1.2.5] 4.1. População prisional: | Medida de segurança - tratamento ambulatorial | Justiça Federal Masculino':'Pop_Prisional_Med_Seg_Just_Fed_Ambulatorial_M',

                        '[Q_4_1.3.5] 4.1. População prisional: | Medida de segurança - tratamento ambulatorial | Justiça Federal Feminino':'Pop_Prisional_Med_Seg_Just_Fed_Ambulatorial_F',

                        '[Q_4_1.4.5] 4.1. População prisional: | Medida de segurança - tratamento ambulatorial | Outros  (Justiça do Trabalho, Cível) Masculino':'Pop_Prisional_Med_Seg_Outros_Ambulatorial_M',

                        '[Q_4_1.5.5] 4.1. População prisional: | Medida de segurança - tratamento ambulatorial | Outros  (Justiça do Trabalho, Cível) Feminino':'Pop_Prisional_Med_Seg_Outros_Ambulatorial_F',

                        '4.1.a. Quantas pessoas privadas de liberdade estão em Regime Disciplinar Diferenciado?':'Qtde_Regime_Disc_Diferenciado',

                        '4.2. O estabelecimento tem controle da informação sobre quantos presos provisórios têm mais de 90 dias de prisão? | Sim. Quantos?':'Sim_Controle_Prov_Mais_90_Dias',

                        '4.2. O estabelecimento tem controle da informação sobre quantos presos provisórios têm mais de 90 dias de prisão? | Sim. Quantos? | Masculino':'Sim_Qtde_Prov_Mais_90_Dias_M',

                        '4.2. O estabelecimento tem controle da informação sobre quantos presos provisórios têm mais de 90 dias de prisão? | Sim. Quantos? | Feminino':'Sim_Qtde_Prov_Mais_90_Dias_F',

                        '4.2. O estabelecimento tem controle da informação sobre quantos presos provisórios têm mais de 90 dias de prisão? | Não':'Nao_Controle_Prov_Mais_90_Dias',

                        '4.2. O estabelecimento tem controle da informação sobre quantos presos provisórios têm mais de 90 dias de prisão? | Não | Masculino':'Nao_Qtde_Prov_Mais_90_Dias_M',

                        '4.2. O estabelecimento tem controle da informação sobre quantos presos provisórios têm mais de 90 dias de prisão? | Não | Feminino':'Nao_Qtde_Prov_Mais_90_Dias_F',

                        '4.3. O estabelecimento tem controle da informação sobre quantos presos sentenciados no regime fechado já foram beneficiados por decisão judicial com o regime semiaberto e aguardam vaga para transferência? | Sim. Quantos?':'Sim_Fechado_Semi_Aguard_Transf',

                        '4.3. O estabelecimento tem controle da informação sobre quantos presos sentenciados no regime fechado já foram beneficiados por decisão judicial com o regime semiaberto e aguardam vaga para transferência? | Sim. Quantos? | Masculino':'Sim_Fechado_Semi_Aguard_Transf_M',

                        '4.3. O estabelecimento tem controle da informação sobre quantos presos sentenciados no regime fechado já foram beneficiados por decisão judicial com o regime semiaberto e aguardam vaga para transferência? | Sim. Quantos? | Feminino':'Sim_Fechado_Semi_Aguard_Transf_F',

                        '4.3. O estabelecimento tem controle da informação sobre quantos presos sentenciados no regime fechado já foram beneficiados por decisão judicial com o regime semiaberto e aguardam vaga para transferência? | Não':'Nao_Fechado_Semi_Aguard_Transf',

                        '4.3. O estabelecimento tem controle da informação sobre quantos presos sentenciados no regime fechado já foram beneficiados por decisão judicial com o regime semiaberto e aguardam vaga para transferência? | Não | Masculino':'Nao_Fechado_Semi_Aguard_Transf_M',

                        '4.3. O estabelecimento tem controle da informação sobre quantos presos sentenciados no regime fechado já foram beneficiados por decisão judicial com o regime semiaberto e aguardam vaga para transferência? | Não | Feminino':'Nao_Fechado_Semi_Aguard_Transf_F',

                        '4.4. O estabelecimento recebe o atestado de pena a cumprir?':'Recebe_Atestado_Pena_A_Cumprir',

                        '4.4.a. Quantas pessoas privadas de liberdade sentenciadas que estão no estabelecimento possuem o atestado de pena atualizado arquivado no prontuário? | Masculino':'Atestado_Pena_Atualizado_Arquivado_M',

                        '4.4.a. Quantas pessoas privadas de liberdade sentenciadas que estão no estabelecimento possuem o atestado de pena atualizado arquivado no prontuário? | Feminino':'Atestado_Pena_Atualizado_Arquivado_F',

                        '4.5.a. Entradas | Número de inclusões originárias (Inclusões não decorrentes de remoção ou transferência de outro estabelecimento do Sistema Prisional) | Masculino':'Inclusoes_Originarias_M',

                        '4.5.a. Entradas | Número de inclusões originárias (Inclusões não decorrentes de remoção ou transferência de outro estabelecimento do Sistema Prisional) | Feminino':'Inclusoes_Originarias_F',

                        '4.5.b. Saídas | Alvarás de soltura (Computar apenas os alvarás que são efetivamente cumpridos, motivando a colocação a pessoa em liberdade) | Masculino':'Alvaras_Soltura_M',

                        '4.5.b. Saídas | Alvarás de soltura (Computar apenas os alvarás que são efetivamente cumpridos, motivando a colocação a pessoa em liberdade) | Feminino':'Alvaras_Soltura_F',

                        '4.5.b. Saídas | Abandonos (Não retorno em saída temporária) | Masculino':'Abandonos_M',

                        '4.5.b. Saídas | Abandonos (Não retorno em saída temporária) | Feminino':'Abandonos_F',

                        '4.5.b. Saídas | Total de óbitos (Independente da causa da mortalidade) | Masculino':'Obitos_M',

                        '4.5.b. Saídas | Total de óbitos (Independente da causa da mortalidade) | Feminino':'Obitos_F',

                        '4.5.c. Transferências/remoções | Número de inclusões por transferências ou remoções (Recebimento de pessoas privadas de liberdade oriundas de outros estabelecimentos do próprio Sistema Prisional) | Masculino':'Entrada_Por_Transferencias_Remocoes_M',

                        '4.5.c. Transferências/remoções | Número de inclusões por transferências ou remoções (Recebimento de pessoas privadas de liberdade oriundas de outros estabelecimentos do próprio Sistema Prisional) | Feminino':'Entrada_Por_Transferencias_Remocoes_F',

                        '4.5.c. Transferências/remoções | Transferências/ remoções - deste para outro estabelecimento | Masculino':'Saida_Por_Transferencias_Remocoes_M',

                        '4.5.c. Transferências/remoções | Transferências/ remoções - deste para outro estabelecimento | Feminino':'Saida_Por_Transferencias_Remocoes_F',

                        '4.5.d. Autorizações de saída | Permissão de saída - para os condenados do regime fechado e semiaberto ou provisórios, por falecimento ou doença grave de parente ou necessidade de tratamento médico (Art. 120, da Lei de Execução Penal) | Masculino':'Qtde_Saida_Falecimento_Doenca_Parente_Ou_Tratamento_Medico_M',

                        '4.5.d. Autorizações de saída | Permissão de saída - para os condenados do regime fechado e semiaberto ou provisórios, por falecimento ou doença grave de parente ou necessidade de tratamento médico (Art. 120, da Lei de Execução Penal) | Feminino':'Qtde_Saida_Falecimento_Doenca_Parente_Ou_Tratamento_Medico_F',

                        '4.5.d. Autorizações de saída | Saída temporária - para os condenados que cumprem pena em regime semiaberto para visitar família (Art. 122, inciso I, da Lei de Execução Penal) | Masculino':'Qtde_Saida_Semi_Visita_Familiar_M',

                        '4.5.d. Autorizações de saída | Saída temporária - para os condenados que cumprem pena em regime semiaberto para visitar família (Art. 122, inciso I, da Lei de Execução Penal) | Feminino':'Qtde_Saida_Semi_Visita_Familiar_F',

                        '5.1. Quantidade de pessoas privadas de liberdade por faixa etária. O estabelecimento tem condições de obter estas informações em seus registros?':'Tem_Info_Sobre_Faixa_Etaria',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 18 a 24 anos | Masculino':'Qtde_Pop_Pris_18_a_24_M',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 18 a 24 anos | Feminino':'Qtde_Pop_Pris_18_a_24_F',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 25 a 29 anos | Masculino':'Qtde_Pop_Pris_25_a_29_M',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 25 a 29 anos | Feminino':'Qtde_Pop_Pris_25_a_29_F',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 30 a 34 anos | Masculino':'Qtde_Pop_Pris_30_a_34_M',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 30 a 34 anos | Feminino':'Qtde_Pop_Pris_30_a_34_F',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 35 a 45 anos | Masculino':'Qtde_Pop_Pris_35_a_45_M',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 35 a 45 anos | Feminino':'Qtde_Pop_Pris_35_a_45_F',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 46 a 60 anos | Masculino':'Qtde_Pop_Pris_46_a_60_M',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 46 a 60 anos | Feminino':'Qtde_Pop_Pris_46_a_60_F',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 61 a 70 anos | Masculino':'Qtde_Pop_Pris_61_a_70_M',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 61 a 70 anos | Feminino':'Qtde_Pop_Pris_61_a_70_F',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 70 anos | Masculino':'Qtde_Pop_Pris_acima_70_M',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 70 anos | Feminino':'Qtde_Pop_Pris_acima_70_F',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Não informado | Masculino':'Qtde_Pop_Pris_idade_Nao_Informado_M',

                        '5.1.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Não informado | Feminino':'Qtde_Pop_Pris_idade_Nao_Informado_F',

                        '5.2. Quantidade de pessoas privadas de liberdade por cor de pele/ raça/ etnia.Para os fins do presente formulário entende-se:      Raça: grupo definido socialmente devido a características físicas, tais como cor de pele, textura do cabelo, traços faciais.      Etnia: grupo definido pelo compartilhamento histórico, religioso ou cultural. As informações devem ser preenchidas de acordo com os registros do estabelecimento, referente às pessoas privadas de liberdade em 30/06/2014. O estabelecimento tem condições de obter estas informações em seus registros?':'Tem_Controle_Por_Etnia',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Branca | Masculino':'Qtde_Brancos_M',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Branca | Feminino':'Qtde__Brancos_F',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Negra | Masculino':'Qtde_Negros_M',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Negra | Feminino':'Qtde_Negros_F',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Parda | Masculino':'Qtde_Pardos_M',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Parda | Feminino':'Qtde_Pardos_F',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Amarela | Masculino':'Qtde_Amarelos_M',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Amarela | Feminino':'Qtde_Amarelos_F',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Indígena | Masculino':'Qtde_Indigenas_M',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Indígena | Feminino':'Qtde_Indigenas_F',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Outras | Masculino':'Qtde_Outros_Racas_M',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Outras | Feminino':'Qtde_Outros_Racas_F',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Não informado | Masculino':'Qtde_Nao_Inf_Racas_M',

                        '5.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Não informado | Feminino':'Qtde_Nao_Inf_Racas_F',

                        '5.2.b. Se houver indígenas, destacar povo indígena ao qual pertence e respectivo idioma: | Povo indígena 1 | Nome do povo indígena':'Nome_Povo_Indigena1',

                        '5.2.b. Se houver indígenas, destacar povo indígena ao qual pertence e respectivo idioma: | Povo indígena 1 | Idioma':'Idioma_Povo_Indigena1',

                        '5.2.b. Se houver indígenas, destacar povo indígena ao qual pertence e respectivo idioma: | Povo indígena 1 | Quantidade':'Qtde_Povo_Indigena1',

                        '5.2.b. Se houver indígenas, destacar povo indígena ao qual pertence e respectivo idioma: | Povo indígena 2 | Nome do povo indígena':'Nome_Povo_Indigena2',

                        '5.2.b. Se houver indígenas, destacar povo indígena ao qual pertence e respectivo idioma: | Povo indígena 2 | Idioma':'Idioma_Povo_Indigena2',

                        '5.2.b. Se houver indígenas, destacar povo indígena ao qual pertence e respectivo idioma: | Povo indígena 2 | Quantidade':'Qtde_Povo_Indigena2',

                        '5.2.b. Se houver indígenas, destacar povo indígena ao qual pertence e respectivo idioma: | Povo indígena 3 | Nome do povo indígena':'Nome_Povo_Indigena3',

                        '5.2.b. Se houver indígenas, destacar povo indígena ao qual pertence e respectivo idioma: | Povo indígena 3 | Idioma':'Idioma_Povo_Indigena3',

                        '5.2.b. Se houver indígenas, destacar povo indígena ao qual pertence e respectivo idioma: | Povo indígena 3 | Quantidade':'Qtde_Povo_Indigena3',

                        '5.2.b. Se houver indígenas, destacar povo indígena ao qual pertence e respectivo idioma: | Povo indígena 4 | Nome do povo indígena':'Nome_Povo_Indigena4',

                        '5.2.b. Se houver indígenas, destacar povo indígena ao qual pertence e respectivo idioma: | Povo indígena 4 | Idioma':'Idioma_Povo_Indigena4',

                        '5.2.b. Se houver indígenas, destacar povo indígena ao qual pertence e respectivo idioma: | Povo indígena 4 | Quantidade':'Qtde_Povo_Indigena4',                        

                        '5.3. Estado civil O estabelecimento tem condições de obter estas informações em seus registros?':'Tem_Controle_Estado_Civil',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Solteiro/a | Masculino':'Qtde_Solteiros_M',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Solteiro/a | Feminino':'Qtde_Solteiros_F',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | União estável/ amasiado | Masculino':'Qtde_Uniao_Estavel_M',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | União estável/ amasiado | Feminino':'Qtde_Uniao_Estavel_F',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Casado/a | Masculino':'Qtde_Casados_M',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Casado/a | Feminino':'Qtde_Casados_F',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Separado/a judicialmente | Masculino':'Qtde_Separados_M',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Separado/a judicialmente | Feminino':'Qtde_Separados_F',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Divorciado/a | Masculino':'Qtde_Divorciados_M',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Divorciado/a | Feminino':'Qtde_Divorciados_F',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Viúvo/a | Masculino':'Qtde_Viuvo_M',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Viúvo/a | Feminino':'Qtde_Viuvo_F',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Não informado | Masculino':'Qtde_Estado_Civil_Nao_Informado_M',

                        '5.3.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Não informado | Feminino':'Qtde_Estado_Civil_Nao_Informado_F',

                        '5.4. Pessoas com deficiência O estabelecimento tem condições de obter estas informações em seus registros?':'Tem_Controle_PCD',

                        '5.4.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Total de pessoas privadas de liberdade com deficiência | Masculino':'Pop_Prisional_PCD_M',

                        '5.4.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Total de pessoas privadas de liberdade com deficiência | Feminino':'Pop_Prisional_PCD_F',

                        '5.4.b. Natureza da deficiência: | Pessoas com deficiência intelectual (apresentam limitações no funcionamento mental, afetando tarefas de comunicação, cuidados pessoais, relacionamento social, segurança, determinação, funções acadêmicas, lazer e trabalho.) | Masculino':'Qtde_PCD_Intelectual_M',

                        '5.4.b. Natureza da deficiência: | Pessoas com deficiência intelectual (apresentam limitações no funcionamento mental, afetando tarefas de comunicação, cuidados pessoais, relacionamento social, segurança, determinação, funções acadêmicas, lazer e trabalho.) | Feminino':'Qtde_PCD_Intelectual_F',

                        '5.4.b. Natureza da deficiência: | Pessoas com deficiência física (apresentam limitação do funcionamento físico-motor; são cadeirantes ou pessoas com deficiência motora, causadas por paralisia cerebral, hemiplegias, lesão medular, amputações ou artropatias.) | Masculino':'Qtde_PCD_Def_Fisica_M',

                        '5.4.b. Natureza da deficiência: | Pessoas com deficiência física (apresentam limitação do funcionamento físico-motor; são cadeirantes ou pessoas com deficiência motora, causadas por paralisia cerebral, hemiplegias, lesão medular, amputações ou artropatias.) | Feminino':'Qtde_PCD_Def_Fisica_F',

                        '5.4.b. Natureza da deficiência: | Quantas pessoas, dentre as informadas acima, são cadeirantes? | Masculino':'Qtde_PCD_Cadeirantes_M',

                        '5.4.b. Natureza da deficiência: | Quantas pessoas, dentre as informadas acima, são cadeirantes? | Feminino':'Qtde_PCD_Cadeirantes_F',

                        '5.4.b. Natureza da deficiência: | Pessoas com deficiência auditiva (apresentam perda total da capacidade auditiva. Perda comprovada da capacidade auditiva entre 95% e 100%.) | Masculino':'Qtde_PCD_Def_Auditivo_M',

                        '5.4.b. Natureza da deficiência: | Pessoas com deficiência auditiva (apresentam perda total da capacidade auditiva. Perda comprovada da capacidade auditiva entre 95% e 100%.) | Feminino':'Qtde_PCD_Def_Auditivo_F',

                        '5.4.b. Natureza da deficiência: | Pessoas com deficiência visual (não possuem a capacidade física de enxergar por total falta de acuidade visual.) | Masculino':'Qtde_PCD_Def_Visual_M',

                        '5.4.b. Natureza da deficiência: | Pessoas com deficiência visual (não possuem a capacidade física de enxergar por total falta de acuidade visual.) | Feminino':'Qtde_PCD_Def_Visual_F',

                        '5.4.b. Natureza da deficiência: | Pessoas com deficiências múltiplas	(apresentam duas ou mais deficiências.) | Masculino':'Qtde_PCD_Def_Mult_M',

                        '5.4.b. Natureza da deficiência: | Pessoas com deficiências múltiplas	(apresentam duas ou mais deficiências.) | Feminino':'Qtde_PCD_Def_Mult_F',

                        '5.5. Quantidade de pessoas privadas de liberdade por grau de instruçãoIdentificar o nível mais elevado de instrução de cada pessoa privada de liberdade em 30/06/2014, de acordo com os registros do estabelecimento. O estabelecimento tem condições de obter estas informações em seus registros?':'Tem_Controle_Grau_Instrucao',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Analfabeto | Masculino':'Qtde_Analfabeto_M',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Analfabeto | Feminino':'Qtde_Analfabeto_F',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Alfabetizado (sem cursos regulares) | Masculino':'Qtde_Alfabetizados_M',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Alfabetizado (sem cursos regulares) | Feminino':'Qtde_Alfabetizados_F',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino Fundamental Incompleto | Masculino':'Qtde_Ens_Fund_Incomp_M',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino Fundamental Incompleto | Feminino':'Qtde_Ens_Fund_Incomp_F',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino Fundamental Completo | Masculino':'Qtde_Ens_Fund_Comp_M',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino Fundamental Completo | Feminino':'Qtde_Ens_Fund_Comp_F',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino Médio Incompleto | Masculino':'Qtde_Ens_Med_Incomp_M',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino Médio Incompleto | Feminino':'Qtde_Ens_Med_Incomp_F',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino Médio Completo | Masculino':'Qtde_Ens_Med_Comp_M',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino Médio Completo | Feminino':'Qtde_Ens_Med_Comp_F',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino Superior Incompleto | Masculino':'Qtde_Ens_Sup_Incomp_M',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino Superior Incompleto | Feminino':'Qtde_Ens_Sup_Incomp_F',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino Superior Completo | Masculino':'Qtde_Ens_Sup_Comp_M',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino Superior Completo | Feminino':'Qtde_Ens_Sup_Comp_F',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino acima de Superior Completo | Masculino':'Qtde_Ens_Acim_Sup_Comp_M',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Ensino acima de Superior Completo | Feminino':'Qtde_Ens_Acim_Sup_Comp_F',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Não Informado | Masculino':'Qtde_Ens_Nao_Inf_M',

                        '5.5.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Não Informado | Feminino':'Qtde_Ens_Nao_Inf_F',

                        '5.6. Número de pessoas privadas de liberdade com documentos pessoaisIdentificar os documentos pessoais arquivados no estabelecimento prisional em 30/06/2014. O estabelecimento possui a documentação física das pessoas privadas de liberdade?':'Tem_Controle_Documentacao_Fisica',

                        '5.6.a. Caso o estabelecimento possua documentação física de pessoas privadas de liberdade, é possível identificar as informações abaixo por tipo de documento?':'Tem_Controle_Documentacao_Fisica_Por_Documento',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Certidão de Nascimento | Masculino':'Qtde_Certidao_Nasc_M',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Certidão de Nascimento | Feminino':'Qtde_Certidao_Nasc_F',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | R.G. | Masculino':'Qtde_RG_M',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | R.G. | Feminino':'Qtde_RG_F',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | C.P.F. | Masculino':'Qtde_CPF_M',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | C.P.F. | Feminino':'Qtde_CPF_F',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Título de eleitor | Masculino':'Qtde_Titulo_Eleitor_M',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Título de eleitor | Feminino':'Qtde_Titulo_Eleitor_F',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Certificado de reservista | Masculino':'Qtde_Reservista_M',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Certificado de reservista | Feminino':'Qtde_Reservista_F',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | CTPS (Carteira de Trabalho) | Masculino':'Qtde_CPTS_M',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | CTPS (Carteira de Trabalho) | Feminino':'Qtde_CPTS_F',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Cartão SUS | Masculino':'Qtde_Cartao_SUS_M',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Cartão SUS | Feminino':'Qtde_Cartao_SUS_F',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | RNE (presos estrangeiros) | Masculino':'Qtde_RNE_M',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | RNE (presos estrangeiros) | Feminino':'Qtde_RNE_F',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Passaporte (presos estrangeiros) | Masculino':'Qtde_Passaporte_Estrangeiros_M',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Passaporte (presos estrangeiros) | Feminino':'Qtde_Passaporte_Estrangeiros_F',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Número de pessoas com algum dos documentos acima | Masculino':'Qtde_Num_Pesooas_Docum_Acima_M',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Número de pessoas com algum dos documentos acima | Feminino':'Qtde_Num_Pesooas_Docum_Acima_F',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Número de pessoas sem documentos | Masculino':'Qtde_Pessoas_Sem_Documento_M',

                        '5.6.b. Em caso positivo, preencha as informações abaixo: | Número de pessoas sem documentos | Feminino':'Qtde_Pessoas_Sem_Documento_F',

                        '5.7. Quantidade de pessoas privadas de liberdade por nacionalidadeIdentificar a nacionalidade das pessoas privadas de liberdade em 30/06/2014, de acordo com os registros do estabelecimento. Se houver dupla nacionalidade e uma das nacionalidades for brasileira, considerar, para os fins do presente formulário, como brasileira. O estabelecimento tem condições de obter estas informações em seus registros?':'Tem_Controle_Dupla_Nacionalidade',

                        '5.7.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Brasileiro nato | Masculino':'Qtde_Brasileiros_Nato_M',

                        '5.7.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Brasileiro nato | Feminino':'Qtde_Brasileiros_Nato_F',

                        '5.7.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Brasileiro naturalizado | Masculino':'Qtde_Brasileiros_Naturalizados_M',

                        '5.7.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Brasileiro naturalizado | Feminino':'Qtde_Brasileiros_Naturalizados_F',

                        '5.7.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Estrangeiros - Total | Masculino':'Qtde_Estrangeiros_M',

                        '5.7.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Estrangeiros - Total | Feminino':'Qtde_Estrangeiros_F',

                        '5.7.b. Estrangeiros - Europa | Alemanha | Masculino':'Qtde_Europa_Alemaes_M',

                        '5.7.b. Estrangeiros - Europa | Alemanha | Feminino':'Qtde_Europa_Alemaes_F',

                        '5.7.b. Estrangeiros - Europa | Áustria | Masculino':'Qtde_Europa_Austriacos_M',

                        '5.7.b. Estrangeiros - Europa | Áustria | Feminino':'Qtde_Europa_Austriacos_F',

                        '5.7.b. Estrangeiros - Europa | Bélgica | Masculino':'Qtde_Europa_Belgas_M',

                        '5.7.b. Estrangeiros - Europa | Bélgica | Feminino':'Qtde_Europa_Europa_Belgas_F',

                        '5.7.b. Estrangeiros - Europa | Bulgária | Masculino':'Qtde_Europa_Bulgaros_M',

                        '5.7.b. Estrangeiros - Europa | Bulgária | Feminino':'Qtde_Europa_Bulgaros_F',

                        '5.7.b. Estrangeiros - Europa | República Tcheca | Masculino':'Qtde_Europa_Tchecos_M',

                        '5.7.b. Estrangeiros - Europa | República Tcheca | Feminino':'Qtde_Europa_Tchecos_F',

                        '5.7.b. Estrangeiros - Europa | Croácia | Masculino':'Qtde_Europa_Croatas_M',

                        '5.7.b. Estrangeiros - Europa | Croácia | Feminino':'Qtde_Europa_Croatas_F',

                        '5.7.b. Estrangeiros - Europa | Dinamarca | Masculino':'Qtde_Europa_Dinamarqueses_M',

                        '5.7.b. Estrangeiros - Europa | Dinamarca | Feminino':'Qtde_Europa_Dinamarqueses_F',

                        '5.7.b. Estrangeiros - Europa | Escócia | Masculino':'Qtde_Europa_Escoceses_M',

                        '5.7.b. Estrangeiros - Europa | Escócia | Feminino':'Qtde_Europa_Escoceses_F',

                        '5.7.b. Estrangeiros - Europa | Espanha | Masculino':'Qtde_Europa_Espanhois_M',

                        '5.7.b. Estrangeiros - Europa | Espanha | Feminino':'Qtde_Europa_Europa_Espanhois_F',

                        '5.7.b. Estrangeiros - Europa | França | Masculino':'Qtde_Europa_Franceses_M',

                        '5.7.b. Estrangeiros - Europa | França | Feminino':'Qtde_Europa_Franceses_F',

                        '5.7.b. Estrangeiros - Europa | Grécia | Masculino':'Qtde_Europa_Europa_Gragos_M',

                        '5.7.b. Estrangeiros - Europa | Grécia | Feminino':'Qtde_Europa_Gragos_F',

                        '5.7.b. Estrangeiros - Europa | Holanda | Masculino':'Qtde_Europa_Holandeses_M',

                        '5.7.b. Estrangeiros - Europa | Holanda | Feminino':'Qtde_Europa_Holandeses_F',

                        '5.7.b. Estrangeiros - Europa | Hungria | Masculino':'Qtde_Europa_Hungria_M',

                        '5.7.b. Estrangeiros - Europa | Hungria | Feminino':'Qtde_Europa_Hungria_F',

                        '5.7.b. Estrangeiros - Europa | Inglaterra | Masculino':'Qtde_Europa_Ingleses_M',

                        '5.7.b. Estrangeiros - Europa | Inglaterra | Feminino':'Qtde_Europa_Ingleses_F',

                        '5.7.b. Estrangeiros - Europa | Irlanda | Masculino':'Qtde_Europa_Irlandeses_M',

                        '5.7.b. Estrangeiros - Europa | Irlanda | Feminino':'Qtde_Europa_Irlandeses_F',

                        '5.7.b. Estrangeiros - Europa | Itália | Masculino':'Qtde_Europa_Italianos_M',

                        '5.7.b. Estrangeiros - Europa | Itália | Feminino':'Qtde_Europa_Italianos_F',

                        '5.7.b. Estrangeiros - Europa | Noruega | Masculino':'Qtde_Europa_Noruega_M',

                        '5.7.b. Estrangeiros - Europa | Noruega | Feminino':'Qtde_Europa_Noruega_F',

                        '5.7.b. Estrangeiros - Europa | País de Gales | Masculino':'Qtde_Europa_Galeses_M',

                        '5.7.b. Estrangeiros - Europa | País de Gales | Feminino':'Qtde_Europa_Galeses_F',

                        '5.7.b. Estrangeiros - Europa | Polônia | Masculino':'Qtde_Europa_Poloneses_M',

                        '5.7.b. Estrangeiros - Europa | Polônia | Feminino':'Qtde_Europa_Poloneses_F',

                        '5.7.b. Estrangeiros - Europa | Portugal | Masculino':'Qtde_Europa_Portugueses_M',

                        '5.7.b. Estrangeiros - Europa | Portugal | Feminino':'Qtde_Europa_Portugueses_F',

                        '5.7.b. Estrangeiros - Europa | Rússia | Masculino':'Qtde_Europa_Russos_M',

                        '5.7.b. Estrangeiros - Europa | Rússia | Feminino':'Qtde_Europa_Russos_F',

                        '5.7.b. Estrangeiros - Europa | Reino Unido | Masculino':'Qtde_Europa_Reino_Unido_M',

                        '5.7.b. Estrangeiros - Europa | Reino Unido | Feminino':'Qtde_Europa_Reino_Unido_F',

                        '5.7.b. Estrangeiros - Europa | Romênia | Masculino':'Qtde_Europa_Romenos_M',

                        '5.7.b. Estrangeiros - Europa | Romênia | Feminino':'Qtde_Europa_Romenos_F',

                        '5.7.b. Estrangeiros - Europa | Sérvia | Masculino':'Qtde_Europa_Servos_M',

                        '5.7.b. Estrangeiros - Europa | Sérvia | Feminino':'Qtde_Europa_Servos_F',

                        '5.7.b. Estrangeiros - Europa | Suécia | Masculino':'Qtde_Europa_Sueco_M',

                        '5.7.b. Estrangeiros - Europa | Suécia | Feminino':'Qtde_Europa_Sueco_F',

                        '5.7.b. Estrangeiros - Europa | Suíça | Masculino':'Qtde_Europa_Suicos_M',

                        '5.7.b. Estrangeiros - Europa | Suíça | Feminino':'Qtde_Europa_Suicos_F',

                        '5.7.b. Estrangeiros - Europa | Outros países do continente europeu | Masculino':'Qtde_Europa_Outros_Paises_M',

                        '5.7.b. Estrangeiros - Europa | Outros países do continente europeu | Feminino':'Qtde_Europa_Outros_Paises_F',

                        '5.7.c. Estrangeiros - Ásia | Afeganistão | Masculino':'Qtde_Asia_Afeganistao_M',

                        '5.7.c. Estrangeiros - Ásia | Afeganistão | Feminino':'Qtde_Asia_Afeganistao_F',

                        '5.7.c. Estrangeiros - Ásia | Arábia Saudita | Masculino':'Qtde_Asia_Arabia_Saudita_M',

                        '5.7.c. Estrangeiros - Ásia | Arábia Saudita | Feminino':'Qtde_Asia_Arabia_Saudita_F',

                        '5.7.c. Estrangeiros - Ásia | Catar | Masculino':'Qtde_Asia_Catar_M',

                        '5.7.c. Estrangeiros - Ásia | Catar | Feminino':'Qtde_Asia_Catar_F',

                        '5.7.c. Estrangeiros - Ásia | Cazaquistão | Masculino':'Qtde_Asia_Cazaquistao_M',

                        '5.7.c. Estrangeiros - Ásia | Cazaquistão | Feminino':'Qtde_Asia_Cazaquistao_F',

                        '5.7.c. Estrangeiros - Ásia | China | Masculino':'Qtde_Asia_China_M',

                        '5.7.c. Estrangeiros - Ásia | China | Feminino':'Qtde_Asia_China_F',

                        '5.7.c. Estrangeiros - Ásia | Coréia do Norte | Masculino':'Qtde_Asia_Coreia_Norte_M',

                        '5.7.c. Estrangeiros - Ásia | Coréia do Norte | Feminino':'Qtde_Asia_Coreia_Norte_F',

                        '5.7.c. Estrangeiros - Ásia | Coréia do Sul | Masculino':'Qtde_Asia_Coreia_Sul_M',

                        '5.7.c. Estrangeiros - Ásia | Coréia do Sul | Feminino':'Qtde_Asia_Coreia_Sul_F',

                        '5.7.c. Estrangeiros - Ásia | Emirados Árabes Unidos | Masculino':'Qtde_Asia_Emirados_Arabes_M',

                        '5.7.c. Estrangeiros - Ásia | Emirados Árabes Unidos | Feminino':'Qtde_Asia_Emirados_Arabes_F',

                        '5.7.c. Estrangeiros - Ásia | Filipinas | Masculino':'Qtde_Asia_Filipinas_M',

                        '5.7.c. Estrangeiros - Ásia | Filipinas | Feminino':'Qtde_Asia_Filipinas_F',

                        '5.7.c. Estrangeiros - Ásia | Índia | Masculino':'Qtde_Asia_India_M',

                        '5.7.c. Estrangeiros - Ásia | Índia | Feminino':'Qtde_Asia_India_F',

                        '5.7.c. Estrangeiros - Ásia | Indonésia | Masculino':'Qtde_Asia_Indonesia_M',

                        '5.7.c. Estrangeiros - Ásia | Indonésia | Feminino':'Qtde_Asia_Indonesia_F',

                        '5.7.c. Estrangeiros - Ásia | Irã | Masculino':'Qtde_Asia_Ira_M',

                        '5.7.c. Estrangeiros - Ásia | Irã | Feminino':'Qtde_Asia_Ira_F',

                        '5.7.c. Estrangeiros - Ásia | Iraque | Masculino':'Qtde_Asia_Iraque_M',

                        '5.7.c. Estrangeiros - Ásia | Iraque | Feminino':'Qtde_Asia_Iraque_F',

                        '5.7.c. Estrangeiros - Ásia | Israel | Masculino':'Qtde_Asia_Irael_M',

                        '5.7.c. Estrangeiros - Ásia | Israel | Feminino':'Qtde_Asia_Irael_F',

                        '5.7.c. Estrangeiros - Ásia | Japão | Masculino':'Qtde_Asia_Japao_M',

                        '5.7.c. Estrangeiros - Ásia | Japão | Feminino':'Qtde_Asia_Japao_F',

                        '5.7.c. Estrangeiros - Ásia | Jordânia | Masculino':'Qtde_Asia_Jordania_M',

                        '5.7.c. Estrangeiros - Ásia | Jordânia | Feminino':'Qtde_Asia_Jordania_F',

                        '5.7.c. Estrangeiros - Ásia | Kuwait | Masculino':'Qtde_Asia_Kuwait_M',

                        '5.7.c. Estrangeiros - Ásia | Kuwait | Feminino':'Qtde_Asia_Kuwait_F',

                        '5.7.c. Estrangeiros - Ásia | Líbano | Masculino':'Qtde_Asia_Libano_M',

                        '5.7.c. Estrangeiros - Ásia | Líbano | Feminino':'Qtde_Asia_Libano_F',

                        '5.7.c. Estrangeiros - Ásia | Macau | Masculino':'Qtde_Asia_Macau_M',

                        '5.7.c. Estrangeiros - Ásia | Macau | Feminino':'Qtde_Asia_Macau_F',

                        '5.7.c. Estrangeiros - Ásia | Malásia | Masculino':'Qtde_Asia_Malasia_M',

                        '5.7.c. Estrangeiros - Ásia | Malásia | Feminino':'Qtde_Asia_Malasia_F',

                        '5.7.c. Estrangeiros - Ásia | Paquistão | Masculino':'Qtde_Asia_Paquistao_M',

                        '5.7.c. Estrangeiros - Ásia | Paquistão | Feminino':'Qtde_Asia_Paquistao_F',

                        '5.7.c. Estrangeiros - Ásia | Síria | Masculino':'Qtde_Asia_Siria_M',

                        '5.7.c. Estrangeiros - Ásia | Síria | Feminino':'Qtde_Asia_Siria_F',

                        '5.7.c. Estrangeiros - Ásia | Sri Lanka | Masculino':'Qtde_Asia_Sri_Lanka_M',

                        '5.7.c. Estrangeiros - Ásia | Sri Lanka | Feminino':'Qtde_Asia_Sri_Lanka_F',

                        '5.7.c. Estrangeiros - Ásia | Tailândia | Masculino':'Qtde_Asia_Tailandia_M',

                        '5.7.c. Estrangeiros - Ásia | Tailândia | Feminino':'Qtde_Asia_Tailandia_F',

                        '5.7.c. Estrangeiros - Ásia | Taiwan | Masculino':'Qtde_Asia_Taiwan_M',

                        '5.7.c. Estrangeiros - Ásia | Taiwan | Feminino':'Qtde_Asia_Taiwan_F',

                        '5.7.c. Estrangeiros - Ásia | Turquia | Masculino':'Qtde_Asia_Turquia_M',

                        '5.7.c. Estrangeiros - Ásia | Turquia | Feminino':'Qtde_Asia_Turquia_F',

                        '5.7.c. Estrangeiros - Ásia | Timor Leste | Masculino':'Qtde_Asia_Timor_Leste_M',

                        '5.7.c. Estrangeiros - Ásia | Timor Leste | Feminino':'Qtde_Asia_Timor_Leste_F',

                        '5.7.c. Estrangeiros - Ásia | Vietnã | Masculino':'Qtde_Asia_Vietna_M',

                        '5.7.c. Estrangeiros - Ásia | Vietnã | Feminino':'Qtde_Asia_Vietna_F',

                        '5.7.c. Estrangeiros - Ásia | Outros países do continente asiático | Masculino':'Qtde_Asia_Outros_Paises_M',

                        '5.7.c. Estrangeiros - Ásia | Outros países do continente asiático | Feminino':'Qtde_Asia_Outros_Paises_F',

                        '5.7.d. Estrangeiros - África | África do Sul | Masculino':'Qtde_Africa_Africa_Do_Sul_M',

                        '5.7.d. Estrangeiros - África | África do Sul | Feminino':'Qtde_Africa_Africa_Do_Sul_F',

                        '5.7.d. Estrangeiros - África | Angola | Masculino':'Qtde_Africa_Angola_M',

                        '5.7.d. Estrangeiros - África | Angola | Feminino':'Qtde_Africa_Angola_F',

                        '5.7.d. Estrangeiros - África | Argélia | Masculino':'Qtde_Africa_Argelia_M',

                        '5.7.d. Estrangeiros - África | Argélia | Feminino':'Qtde_Africa_Argelia_F',

                        '5.7.d. Estrangeiros - África | Cabo Verde | Masculino':'Qtde_Africa_Cabo_Verde_M',

                        '5.7.d. Estrangeiros - África | Cabo Verde | Feminino':'Qtde_Africa_Cabo_Verde_F',

                        '5.7.d. Estrangeiros - África | Camarões | Masculino':'Qtde_Africa_Camaroes_M',

                        '5.7.d. Estrangeiros - África | Camarões | Feminino':'Qtde_Africa_Camaroes_F',

                        '5.7.d. Estrangeiros - África | República do Congo | Masculino':'Qtde_Africa_Republica_Do_Congo_M',

                        '5.7.d. Estrangeiros - África | República do Congo | Feminino':'Qtde_Africa_Republica_Do_Congo_F',

                        '5.7.d. Estrangeiros - África | Costa do Marfim | Masculino':'Qtde_Africa_Costa_Do_Marfim_M',

                        '5.7.d. Estrangeiros - África | Costa do Marfim | Feminino':'Qtde_Africa_Costa_Do_Marfim_F',

                        '5.7.d. Estrangeiros - África | Egito | Masculino':'Qtde_Africa_Egito_M',

                        '5.7.d. Estrangeiros - África | Egito | Feminino':'Qtde_Africa_Egito_F',

                        '5.7.d. Estrangeiros - África | Etiópia | Masculino':'Qtde_Africa_Etiopia_M',

                        '5.7.d. Estrangeiros - África | Etiópia | Feminino':'Qtde_Africa_Etiopia_F',

                        '5.7.d. Estrangeiros - África | Gana | Masculino':'Qtde_Africa_Gana_M',

                        '5.7.d. Estrangeiros - África | Gana | Feminino':'Qtde_Africa_Gana_F',

                        '5.7.d. Estrangeiros - África | Guiné | Masculino':'Qtde_Africa_Guine_M',

                        '5.7.d. Estrangeiros - África | Guiné | Feminino':'Qtde_Africa_Guine_F',

                        '5.7.d. Estrangeiros - África | Guiné Bissau | Masculino':'Qtde_Africa_Guine_Bissau_M',

                        '5.7.d. Estrangeiros - África | Guiné Bissau | Feminino':'Qtde_Africa_Guine_Bissau_F',

                        '5.7.d. Estrangeiros - África | Líbia | Masculino':'Qtde_Africa_Libia_M',

                        '5.7.d. Estrangeiros - África | Líbia | Feminino':'Qtde_Africa_Libia_F',

                        '5.7.d. Estrangeiros - África | Madagascar | Masculino':'Qtde_Africa_Madagascar_M',

                        '5.7.d. Estrangeiros - África | Madagascar | Feminino':'Qtde_Africa_Madagascar_F',

                        '5.7.d. Estrangeiros - África | Marrocos | Masculino':'Qtde_Africa_Marrocos_M',

                        '5.7.d. Estrangeiros - África | Marrocos | Feminino':'Qtde_Africa_Marrocos_F',

                        '5.7.d. Estrangeiros - África | Moçambique | Masculino':'Qtde_Africa_Mocambique_M',

                        '5.7.d. Estrangeiros - África | Moçambique | Feminino':'Qtde_Africa_Mocambique_F',

                        '5.7.d. Estrangeiros - África | Nigéria | Masculino':'Qtde_Africa_Nigeria_M',

                        '5.7.d. Estrangeiros - África | Nigéria | Feminino':'Qtde_Africa_Nigeria_F',

                        '5.7.d. Estrangeiros - África | Quênia | Masculino':'Qtde_Africa_Quenia_M',

                        '5.7.d. Estrangeiros - África | Quênia | Feminino':'Qtde_Africa_Quenia_F',

                        '5.7.d. Estrangeiros - África | Ruanda | Masculino':'Qtde_Africa_Ruanda_M',

                        '5.7.d. Estrangeiros - África | Ruanda | Feminino':'Qtde_Africa_Ruanda_F',

                        '5.7.d. Estrangeiros - África | Senegal | Masculino':'Qtde_Africa_Senegal_M',

                        '5.7.d. Estrangeiros - África | Senegal | Feminino':'Qtde_Africa_Senegal_F',

                        '5.7.d. Estrangeiros - África | Serra Leoa | Masculino':'Qtde_Africa_Serra_Leoa_M',

                        '5.7.d. Estrangeiros - África | Serra Leoa | Feminino':'Qtde_Africa_Serra_Leoa_F',

                        '5.7.d. Estrangeiros - África | Somália | Masculino':'Qtde_Africa_Somalia_M',

                        '5.7.d. Estrangeiros - África | Somália | Feminino':'Qtde_Africa_Somalia_F',

                        '5.7.d. Estrangeiros - África | Tunísia | Masculino':'Qtde_Africa_Tunisia_M',

                        '5.7.d. Estrangeiros - África | Tunísia | Feminino':'Qtde_Africa_Tunisia_F',

                        '5.7.d. Estrangeiros - África | Outros países do continente africano | Masculino':'Qtde_Africa_Outros_Paises_M',

                        '5.7.d. Estrangeiros - África | Outros países do continente africano | Feminino':'Qtde_Africa_Outros_Paises_F',

                        '5.7.e. Estrangeiros - América | Argentina | Masculino':'Qtde_America_Argentina_M',

                        '5.7.e. Estrangeiros - América | Argentina | Feminino':'Qtde_America_Argentina_F',

                        '5.7.e. Estrangeiros - América | Bolívia | Masculino':'Qtde_America_Bolivia_M',

                        '5.7.e. Estrangeiros - América | Bolívia | Feminino':'Qtde_America_Bolivia_F',

                        '5.7.e. Estrangeiros - América | Canadá | Masculino':'Qtde_America_Canada_M',

                        '5.7.e. Estrangeiros - América | Canadá | Feminino':'Qtde_America_Canada_F',

                        '5.7.e. Estrangeiros - América | Chile | Masculino':'Qtde_America_Chile_M',

                        '5.7.e. Estrangeiros - América | Chile | Feminino':'Qtde_America_Chile_F',

                        '5.7.e. Estrangeiros - América | Colômbia | Masculino':'Qtde_America_Colombia_M',

                        '5.7.e. Estrangeiros - América | Colômbia | Feminino':'Qtde_America_Colombia_F',

                        '5.7.e. Estrangeiros - América | Costa Rica | Masculino':'Qtde_America_Costa_Rica_M',

                        '5.7.e. Estrangeiros - América | Costa Rica | Feminino':'Qtde_America_Costa_Rica_F',

                        '5.7.e. Estrangeiros - América | Cuba | Masculino':'Qtde_America_Cuba_M',

                        '5.7.e. Estrangeiros - América | Cuba | Feminino':'Qtde_America_Cuba_F',

                        '5.7.e. Estrangeiros - América | El Salvador | Masculino':'Qtde_America_El_Salvador_M',

                        '5.7.e. Estrangeiros - América | El Salvador | Feminino':'Qtde_America_El_Salvador_F',

                        '5.7.e. Estrangeiros - América | Equador | Masculino':'Qtde_America_Equador_M',

                        '5.7.e. Estrangeiros - América | Equador | Feminino':'Qtde_America_Equador_F',

                        '5.7.e. Estrangeiros - América | Estados Unidos da América | Masculino':'Qtde_America_EUA_M',

                        '5.7.e. Estrangeiros - América | Estados Unidos da América | Feminino':'Qtde_America_EUA_F',

                        '5.7.e. Estrangeiros - América | Guatemala | Masculino':'Qtde_America_Quatemala_M',

                        '5.7.e. Estrangeiros - América | Guatemala | Feminino':'Qtde_America_Quatemala_F',

                        '5.7.e. Estrangeiros - América | Guiana | Masculino':'Qtde_America_Quiana_M',

                        '5.7.e. Estrangeiros - América | Guiana | Feminino':'Qtde_America_Quiana_F',

                        '5.7.e. Estrangeiros - América | Guiana Francesa | Masculino':'Qtde_America_Quiana_Francesa_M',

                        '5.7.e. Estrangeiros - América | Guiana Francesa | Feminino':'Qtde_America_Quiana_Francesa_F',

                        '5.7.e. Estrangeiros - América | Haiti | Masculino':'Qtde_America_Haiti_M',

                        '5.7.e. Estrangeiros - América | Haiti | Feminino':'Qtde_America_Haiti_F',

                        '5.7.e. Estrangeiros - América | Honduras | Masculino':'Qtde_America_Honduras_M',

                        '5.7.e. Estrangeiros - América | Honduras | Feminino':'Qtde_America_Honduras_F',

                        '5.7.e. Estrangeiros - América | Ilhas Cayman | Masculino':'Qtde_America_Ilhas_Cayman_M',

                        '5.7.e. Estrangeiros - América | Ilhas Cayman | Feminino':'Qtde_America_Ilhas_Cayman_F',

                        '5.7.e. Estrangeiros - América | Jamaica | Masculino':'Qtde_America_Jamaica_M',

                        '5.7.e. Estrangeiros - América | Jamaica | Feminino':'Qtde_America_Jamaica_F',

                        '5.7.e. Estrangeiros - América | México | Masculino':'Qtde_America_Mexico_M',

                        '5.7.e. Estrangeiros - América | México | Feminino':'Qtde_America_Mexico_F',

                        '5.7.e. Estrangeiros - América | Nicarágua | Masculino':'Qtde_America_Nicaragua_M',

                        '5.7.e. Estrangeiros - América | Nicarágua | Feminino':'Qtde_America_Nicaragua_F',

                        '5.7.e. Estrangeiros - América | Panamá | Masculino':'Qtde_America_Panama_M',

                        '5.7.e. Estrangeiros - América | Panamá | Feminino':'Qtde_America_Panama_F',

                        '5.7.e. Estrangeiros - América | Paraguai | Masculino':'Qtde_America_Paraguai_M',

                        '5.7.e. Estrangeiros - América | Paraguai | Feminino':'Qtde_America_Paraguai_F',

                        '5.7.e. Estrangeiros - América | Peru | Masculino':'Qtde_America_Peru_M',

                        '5.7.e. Estrangeiros - América | Peru | Feminino':'Qtde_America_Peru_F',

                        '5.7.e. Estrangeiros - América | Porto Rico | Masculino':'Qtde_America_Porto_Rico_M',

                        '5.7.e. Estrangeiros - América | Porto Rico | Feminino':'Qtde_America_Porto_Rico_F',

                        '5.7.e. Estrangeiros - América | República Dominicana | Masculino':'Qtde_America_Republica_Dominicana_M',

                        '5.7.e. Estrangeiros - América | República Dominicana | Feminino':'Qtde_America_Republica_Dominicana_F',

                        '5.7.e. Estrangeiros - América | Suriname | Masculino':'Qtde_America_Suriname_M',

                        '5.7.e. Estrangeiros - América | Suriname | Feminino':'Qtde_America_Suriname_F',

                        '5.7.e. Estrangeiros - América | Trindade e Tobago | Masculino':'Qtde_America_Trinidad_Tobago_M',

                        '5.7.e. Estrangeiros - América | Trindade e Tobago | Feminino':'Qtde_America_Trinidad_Tobago_F',

                        '5.7.e. Estrangeiros - América | Uruguai | Masculino':'Qtde_America_Uruguai_M',

                        '5.7.e. Estrangeiros - América | Uruguai | Feminino':'Qtde_America_Uruguai_F',

                        '5.7.e. Estrangeiros - América | Venezuela | Masculino':'Qtde_America_Venezuela_M',

                        '5.7.e. Estrangeiros - América | Venezuela | Feminino':'Qtde_America_Venezuela_F',

                        '5.7.e. Estrangeiros - América | Outros países do continente americano | Masculino':'Qtde_America_Outros_Paises_M',

                        '5.7.e. Estrangeiros - América | Outros países do continente americano | Feminino':'Qtde_America_Outros_Paises_F',

                        '5.7.f. Estrangeiros - Oceania | Austrália | Masculino':'Qtde_Oceania_Australia_M',

                        '5.7.f. Estrangeiros - Oceania | Austrália | Feminino':'Qtde_Oceania_Australia_F',

                        '5.7.f. Estrangeiros - Oceania | Nova Zelândia | Masculino':'Qtde_Oceania_Nova_Zelandia_M',

                        '5.7.f. Estrangeiros - Oceania | Nova Zelândia | Feminino':'Qtde_Oceania_Nova_Zelandia_F',

                        '5.7.f. Estrangeiros - Oceania | Outros países do continente Oceania | Masculino':'Qtde_Oceania_Outros_Paises_M',

                        '5.7.f. Estrangeiros - Oceania | Outros países do continente Oceania | Feminino':'Qtde_Oceania_Outros_Paises_F',

                        '5.7.g. Sem informação | Número de pessoas sem informação sobre nacionalidade | Masculino':'Qtde_Sem_Nacionalidade_M',

                        '5.7.g. Sem informação | Número de pessoas sem informação sobre nacionalidade | Feminino':'Qtde_Sem_Nacionalidade_F',

                        '5.8. Faixa etária dos filhos que estão no estabelecimento (aplicável apenas para estabelecimentos com mulheres) | 0 a 6 meses | Quantidade por faixa etária':'Qtde_Filhos_0_6_Meses',

                        '5.8. Faixa etária dos filhos que estão no estabelecimento (aplicável apenas para estabelecimentos com mulheres) | mais de 6 meses a 1 ano | Quantidade por faixa etária':'Qtde_Filhos_6_Meses_1_Ano',

                        '5.8. Faixa etária dos filhos que estão no estabelecimento (aplicável apenas para estabelecimentos com mulheres) | mais de 1 ano a 2 anos | Quantidade por faixa etária':'Qtde_Filhos_1_2_Ano',

                        '5.8. Faixa etária dos filhos que estão no estabelecimento (aplicável apenas para estabelecimentos com mulheres) | mais de 2 a 3 anos | Quantidade por faixa etária':'Qtde_Filhos_2_3_Ano',

                        '5.8. Faixa etária dos filhos que estão no estabelecimento (aplicável apenas para estabelecimentos com mulheres) | mais de 3 anos | Quantidade por faixa etária':'Qtde_Filhos_mais_3_Ano',

                        '5.9. Número de filhos/as. O estabelecimento detém alguma forma de registro que permite a obtenção desta informação?':'Tem_Controle_Sobre_Num_Filhos',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Sem filhos | Masculino':'Qtde_Pop_Prisional_Sem_Filhos_M',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Sem filhos | Feminino':'Qtde_Pop_Prisional_Sem_Filhos_F',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 1 | Masculino':'Qtde_Pop_Prisional_1_Filho_M',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 1 | Feminino':'Qtde_Pop_Prisional_1_Filho_F',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 2 | Masculino':'Qtde_Pop_Prisional_2_Filhos_M',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 2 | Feminino':'Qtde_Pop_Prisional_2_Filhos_F',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 3 | Masculino':'Qtde_Pop_Prisional_3_Filhos_M',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 3 | Feminino':'Qtde_Pop_Prisional_3_Filhos_F',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 4 | Masculino':'Qtde_Pop_Prisional_4_Filhos_M',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 4 | Feminino':'Qtde_Pop_Prisional_4_Filhos_F',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 5 | Masculino':'Qtde_Pop_Prisional_5_Filhos_M',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 5 | Feminino':'Qtde_Pop_Prisional_5_Filhos_F',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 6 | Masculino':'Qtde_Pop_Prisional_6_Filhos_M',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 6 | Feminino':'Qtde_Pop_Prisional_6_Filhos_F',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 7 | Masculino':'Qtde_Pop_Prisional_7_Filhos_M',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 7 | Feminino':'Qtde_Pop_Prisional_7_Filhos_F',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 8 | Masculino':'Qtde_Pop_Prisional_8_Filhos_M',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 8 | Feminino':'Qtde_Pop_Prisional_8_Filhos_F',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 9 | Masculino':'Qtde_Pop_Prisional_9_Filhos_M',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 9 | Feminino':'Qtde_Pop_Prisional_9_Filhos_F',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 10 | Masculino':'Qtde_Pop_Prisional_10_Filhos_M',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 10 | Feminino':'Qtde_Pop_Prisional_10_Filhos_F',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 11 ou mais | Masculino':'Qtde_Pop_Prisional_11_Ou_Mais_Filhos_M',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | 11 ou mais | Feminino':'Qtde_Pop_Prisional_11_Ou_Mais_Filhos_F',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Sem informação | Masculino':'Qtde_Pop_Prisional_Sem_Inf_Filhos_M',

                        '5.9.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Sem informação | Feminino':'Qtde_Pop_Prisional_Sem_Inf_Filhos_F',

                        '5.10. Número de pessoas privadas de liberdade que possuem visitantes cadastrados: | Pessoas com visitantes cadastrados | Masculino':'Qtde_Pop_Prisional_Com_Visit_Cadast_M',

                        '5.10. Número de pessoas privadas de liberdade que possuem visitantes cadastrados: | Pessoas com visitantes cadastrados | Feminino':'Qtde_Pop_Prisional_Com_Visit_Cadast_F',

                        '5.11. Quantidade de pessoas privadas de liberdade por tempo total de penas (presos/as condenados/as). O estabelecimento detém alguma forma de registro que permite a obtenção desta informação?':'Tem_Controle_Tempo_Total_Penas',

                        '5.11.a. Como é registrada essa informação?':'Forma_Registro_Total_Penas',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Até 6 meses (inclusive) | Masculino':'Qtde_Penas_Ate_6_Meses_M',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Até 6 meses (inclusive) | Feminino':'Qtde_Penas_Ate_6_Meses_F',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 6 meses até 1 ano (inclusive) | Masculino':'Qtde_Penas_6_Meses_1_Ano_M',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 6 meses até 1 ano (inclusive) | Feminino':'Qtde_Penas_6_Meses_1_Ano_F',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 1 ano até 2 anos (inclusive) | Masculino':'Qtde_Penas_1_2_Anos_M',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 1 ano até 2 anos (inclusive) | Feminino':'Qtde_Penas_1_2_Anos_F',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 2 até 4 anos (inclusive) | Masculino':'Qtde_Penas_2_4_Anos_M',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 2 até 4 anos (inclusive) | Feminino':'Qtde_Penas_2_4_Anos_F',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 4 até 8 anos (inclusive) | Masculino':'Qtde_Penas_4_8_Anos_M',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 4 até 8 anos (inclusive) | Feminino':'Qtde_Penas_4_8_Anos_F',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 8 até 15 anos (inclusive) | Masculino':'Qtde_Penas_8_15_Anos_M',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 8 até 15 anos (inclusive) | Feminino':'Qtde_Penas_8_15_Anos_F',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 15 até 20 anos (inclusive) | Masculino':'Qtde_Penas_15_20_Anos_M',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 15 até 20 anos (inclusive) | Feminino':'Qtde_Penas_15_20_Anos_F',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 20 até 30 anos (inclusive) | Masculino':'Qtde_Penas_20_30_Anos_M',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 20 até 30 anos (inclusive) | Feminino':'Qtde_Penas_20_30_Anos_F',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 30 até 50 anos (inclusive) | Masculino':'Qtde_Penas_30_50_Anos_M',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 30 até 50 anos (inclusive) | Feminino':'Qtde_Penas_30_50_Anos_F',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 50 até 100 anos (inclusive) | Masculino':'Qtde_Penas_50_100_Anos_M',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 50 até 100 anos (inclusive) | Feminino':'Qtde_Penas_50_100_Anos_F',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 100 anos | Masculino':'Qtde_Penas_Mais_100_Anos_M',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 100 anos | Feminino':'Qtde_Penas_Mais_100_Anos_F',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Número de pessoas sem informação | Masculino':'Qtde_Penas_Sem_Inf_Anos_M',

                        '5.11.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Número de pessoas sem informação | Feminino':'Qtde_Penas_Sem_Inf_Anos_F',

                        '5.12. Quantidade de pessoas privadas de liberdade por tempo de pena remanescente (presos/as condenados/as). O estabelecimento detém alguma forma de registro que permite a obtenção desta informação?':'Tem_Controle_Penas_Remanescentes',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Até 6 meses (inclusive) | Masculino':'Qtde_Pop_Prisional_Penas_Remanescentes_0_6_Meses_M',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Até 6 meses (inclusive) | Feminino':'Qtde_Pop_Prisional_Penas_Remanescentes_0_6_Meses_F',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 6 meses até 1 ano (inclusive) | Masculino':'Qtde_Pop_Prisional_Penas_Remanescentes_6_Meses_1_Ano_M',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 6 meses até 1 ano (inclusive) | Feminino':'Qtde_Pop_Prisional_Penas_Remanescentes_6_Meses_1_Ano_F',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 1 ano até 2 anos (inclusive) | Masculino':'Qtde_Pop_Prisional_Penas_Remanescentes_1_2_Anos_M',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 1 ano até 2 anos (inclusive) | Feminino':'Qtde_Pop_Prisional_Penas_Remanescentes_1_2_Anos_F',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 2 até 4 anos (inclusive) | Masculino':'Qtde_Pop_Prisional_Penas_Remanescentes_2_4_Anos_M',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 2 até 4 anos (inclusive) | Feminino':'Qtde_Pop_Prisional_Penas_Remanescentes_2_4_Anos_F',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 4 até 8 anos (inclusive) | Masculino':'Qtde_Pop_Prisional_Penas_Remanescentes_4_8_Anos_M',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 4 até 8 anos (inclusive) | Feminino':'Qtde_Pop_Prisional_Penas_Remanescentes_4_8_Anos_F',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 8 até 15 anos (inclusive) | Masculino':'Qtde_Pop_Prisional_Penas_Remanescentes_8_15_Anos_M',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 8 até 15 anos (inclusive) | Feminino':'Qtde_Pop_Prisional_Penas_Remanescentes_8_15_Anos_F',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 15 até 20 anos (inclusive) | Masculino':'Qtde_Pop_Prisional_Penas_Remanescentes_15_20_Anos_M',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 15 até 20 anos (inclusive) | Feminino':'Qtde_Pop_Prisional_Penas_Remanescentes_15_20_Anos_F',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 20 até 30 anos (inclusive) | Masculino':'Qtde_Pop_Prisional_Penas_Remanescentes_20_30_Anos_M',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 20 até 30 anos (inclusive) | Feminino':'Qtde_Pop_Prisional_Penas_Remanescentes_20_30_Anos_F',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 30 até 50 anos (inclusive) | Masculino':'Qtde_Pop_Prisional_Penas_Remanescentes_30_50_Anos_M',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 30 até 50 anos (inclusive) | Feminino':'Qtde_Pop_Prisional_Penas_Remanescentes_30_50_Anos_F',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 50 até 100 anos (inclusive) | Masculino':'Qtde_Pop_Prisional_Penas_Remanescentes_50_100_Anos_M',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 50 até 100 anos (inclusive) | Feminino':'Qtde_Pop_Prisional_Penas_Remanescentes_50_100_Anos_F',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 100 anos | Masculino':'Qtde_Pop_Prisional_Penas_Remanescentes_Mais_100_Anos_M',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais de 100 anos | Feminino':'Qtde_Pop_Prisional_Penas_Remanescentes_Mais_100_Anos_F',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Número de pessoas sem informação | Masculino':'Qtde_Pop_Prisional_Penas_Remanescentes_Sem_Inf_Anos_M',

                        '5.12.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Número de pessoas sem informação | Feminino':'Qtde_Pop_Prisional_Penas_Remanescentes_Sem_Inf_Anos_F',

                        '5.13. Quantidade de incidências por tipo penal. O estabelecimento detém alguma forma de registro que permite a obtenção desta informação?':'Tem_Controle_Tipo_Penal',

                        '5.13.a. Como é registrada essa informação?':'Forma_Registro_Tipo_Penal',

                        '5.13.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Total | Masculino':'Qtde_Tipo_Penal_Total_M',

                        '5.13.b. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Total | Feminino':'Qtde_Tipo_Penal_Total_F',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Homicídio simples (Art. 121, caput) | Masculino':'Qtde_Homicidios_Simples_M',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Homicídio simples (Art. 121, caput) | Feminino':'Qtde_Homicidios_Simples_F',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Homicílio culposo (Art. 121, § 3°) | Masculino':'Qtde_Homicidios_Culposo_M',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Homicílio culposo (Art. 121, § 3°) | Feminino':'Qtde_Homicidios_Culposo_F',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Homicídio qualificado (Art. 121, § 2°) | Masculino':'Qtde_Homicidios_Qualificado_M',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Homicídio qualificado (Art. 121, § 2°) | Feminino':'Qtde_Homicidios_Qualificado_F',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Aborto (Art. 124, 125, 126 e 127) | Masculino':'Qtde_Aborto_M',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Aborto (Art. 124, 125, 126 e 127) | Feminino':'Qtde_Aborto_F',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Lesão corporal (Art. 129, caput e § 1°, 2°, 3° e 6°) | Masculino':'Qtde_Lesao_Corporal_M',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Lesão corporal (Art. 129, caput e § 1°, 2°, 3° e 6°) | Feminino':'Qtde_Lesao_Corporal_F',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Violência doméstica (Art. 129,  § 9°) | Masculino':'Qtde_Violencia_Domestica_M',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Violência doméstica (Art. 129,  § 9°) | Feminino':'Qtde_Violencia_Domestica_F',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Sequestro e cárcere privado (Art. 148) | Masculino':'Qtde_Sequestro_M',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Sequestro e cárcere privado (Art. 148) | Feminino':'Qtde_Sequestro_F',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Outros - não listados acima entre os artigos 122 e 154-A | Masculino':'Qtde_Outro_Tipo_Penal_M',

                        '5.13.b.1.           Grupo: Crimes contra a pessoa | Outros - não listados acima entre os artigos 122 e 154-A | Feminino':'Qtde_Outro_Tipo_Penal_F',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Furto simples (Art. 155) | Masculino':'Qtde_Furto_Simples_M',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Furto simples (Art. 155) | Feminino':'Qtde_Furto_Simples_F',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Furto qualificado (Art. 155, § 4° e 5°) | Masculino':'Qtde_Furto_Qualificado_M',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Furto qualificado (Art. 155, § 4° e 5°) | Feminino':'Qtde_Furto_Qualificado_F',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Roubo simples (Art. 157) | Masculino':'Qtde_Roubo_Simples_M',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Roubo simples (Art. 157) | Feminino':'Qtde_Roubo_Simples_F',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Roubo qualificado (Art. 157, § 2° | Masculino':'Qtde_Roubo_Qualificado_M',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Roubo qualificado (Art. 157, § 2° | Feminino':'Qtde_Roubo_Qualificado_F',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Latrocínio (Art. 157, § 3°) | Masculino':'Qtde_Latrocinio_M',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Latrocínio (Art. 157, § 3°) | Feminino':'Qtde_Latrocinio_F',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Extorsão (Art. 158) | Masculino':'Qtde_Extorsao_M',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Extorsão (Art. 158) | Feminino':'Qtde_Extorsao_F',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Extorsão mediante sequestro (Art. 159) | Masculino':'Qtde_Extorsao_Mediante_Sequestro_M',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Extorsão mediante sequestro (Art. 159) | Feminino':'Qtde_Extorsao_Mediante_Sequestro_F',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Apropriação indébita (Art. 168) | Masculino':'Qtde_Apropriacao_Indebita_M',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Apropriação indébita (Art. 168) | Feminino':'Qtde_Apropriacao_Indebita_F',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Apropriação indébita previdenciária (Art. 168-A) | Masculino':'Qtde_Apropriacao_Indebita_Previdenciaria_M',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Apropriação indébita previdenciária (Art. 168-A) | Feminino':'Qtde_Apropriacao_Indebita_Previdenciaria_F',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Estelionato (Art. 171) | Masculino':'Qtde_Estelionato_M',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Estelionato (Art. 171) | Feminino':'Qtde_Estelionato_F',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Receptação (Art. 180) | Masculino':'Qtde_Receptacao_M',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Receptação (Art. 180) | Feminino':'Qtde_Receptacao_F',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Receptação qualificada (Art. 180, § 1°) | Masculino':'Qtde_Receptacao_Qualificada_M',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Receptação qualificada (Art. 180, § 1°) | Feminino':'Qtde_Receptacao_Qualificada_F',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Outros - não listados acima entre os artigos 156 e 179 | Masculino':'Qtde_Crimes_Ao_Patrimonio_Outros_M',

                        '5.13.b.2.                     Grupo: Crimes contra o patrimônio | Outros - não listados acima entre os artigos 156 e 179 | Feminino':'Qtde_Crimes_Ao_Patrimonio_Outros_F',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Estupro (Art. 213) | Masculino':'Qtde_Estupro_M',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Estupro (Art. 213) | Feminino':'Qtde_Estupro_F',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Atentado violento ao pudor (Art. 214) | Masculino':'Qtde_Atentado_Violento_Ao_Pudor_M',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Atentado violento ao pudor (Art. 214) | Feminino':'Qtde_Atentado_Violento_Ao_Pudor_F',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Estupro de vulnerável (Art. 217-A) | Masculino':'Qtde_Estupro_De_Vulneravel_M',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Estupro de vulnerável (Art. 217-A) | Feminino':'Qtde_Estupro_De_Vulneravel_F',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Corrupção de menores (Art. 218) | Masculino':'Qtde_Corrupcao_Menores_M',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Corrupção de menores (Art. 218) | Feminino':'Qtde_Corrupcao_Menores_F',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Tráfico internacional de pessoa para fim de exploração sexual (Art. 231) | Masculino':'Qtde_Trafico_Internacional_Exploracao_Sexual_M',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Tráfico internacional de pessoa para fim de exploração sexual (Art. 231) | Feminino':'Qtde_Trafico_Internacional_Exploracao_Sexual_F',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Tráfico interno de pessoa para fim de exploração sexual (Art. 231-A) | Masculino':'Qtde_Trafico_Interno_Exploracao_Sexual_M',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Tráfico interno de pessoa para fim de exploração sexual (Art. 231-A) | Feminino':'Qtde_Trafico_Interno_Exploracao_Sexual_F',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Outros (Artigos 215, 216-A, 218-A, 218-B, 227, 228, 229, 230) | Masculino':'Qtde_Crimes_Contra_Dignidade_Sexual_Outros_M',

                        '5.13.b.3.                               Grupo: Crimes contra a dignidade sexual | Outros (Artigos 215, 216-A, 218-A, 218-B, 227, 228, 229, 230) | Feminino':'Qtde_Crimes_Contra_Dignidade_Sexual_Outros_F',

                        '5.13.b.4.                                         Grupo: Crimes contra a paz pública | Quadrilha ou bando (Art. 288) | Masculino':'Qtde_Quadrilha_Bando_M',

                        '5.13.b.4.                                         Grupo: Crimes contra a paz pública | Quadrilha ou bando (Art. 288) | Feminino':'Qtde_Quadrilha_Bando_F',

                        '5.13.b.5.                                                   Grupo: Crimes contra a fé pública | Moeda falsa (Art. 289) | Masculino':'Qtde_Moeda_Falsa_M',

                        '5.13.b.5.                                                   Grupo: Crimes contra a fé pública | Moeda falsa (Art. 289) | Feminino':'Qtde_Moeda_Falsa_F',

                        '5.13.b.5.                                                   Grupo: Crimes contra a fé pública | Falsificação de papéis, selos, sinal e documentos públicos ( Art. 293 a 297) | Masculino':'Qtde_Falsificacao_Doc_Publicos_M',

                        '5.13.b.5.                                                   Grupo: Crimes contra a fé pública | Falsificação de papéis, selos, sinal e documentos públicos ( Art. 293 a 297) | Feminino':'Qtde_Falsificacao_Doc_Publicos_F',

                        '5.13.b.5.                                                   Grupo: Crimes contra a fé pública | Falsidade ideológica (Art. 299) | Masculino':'Qtde_Falsidade_Ideologica_M',

                        '5.13.b.5.                                                   Grupo: Crimes contra a fé pública | Falsidade ideológica (Art. 299) | Feminino':'Qtde_Falsidade_Ideologica_F',

                        '5.13.b.5.                                                   Grupo: Crimes contra a fé pública | Uso de documento falso (Art. 304) | Masculino':'Qtde_Uso_Doc_Falso_M',

                        '5.13.b.5.                                                   Grupo: Crimes contra a fé pública | Uso de documento falso (Art. 304) | Feminino':'Qtde_Uso_Doc_Falso_F',

                        '5.13.b.6.                                                             Grupo: Crimes contra a Administração Pública | Peculato (Art. 312 e 313) | Masculino':'Qtde_Peculato_M',

                        '5.13.b.6.                                                             Grupo: Crimes contra a Administração Pública | Peculato (Art. 312 e 313) | Feminino':'Qtde_Peculato_F',

                        '5.13.b.6.                                                             Grupo: Crimes contra a Administração Pública | Concussão e excesso de exação (Art. 316) | Masculino':'Qtde_Concussao_M',

                        '5.13.b.6.                                                             Grupo: Crimes contra a Administração Pública | Concussão e excesso de exação (Art. 316) | Feminino':'Qtde_Concussao_F',

                        '5.13.b.6.                                                             Grupo: Crimes contra a Administração Pública | Corrupção passiva (Art. 317) | Masculino':'Qtde_Corrupcao_Passiva_M',

                        '5.13.b.6.                                                             Grupo: Crimes contra a Administração Pública | Corrupção passiva (Art. 317) | Feminino':'Qtde_Corrupcao_Passiva_F',

                        '5.13.b.7.                                                             Grupo: Crimes praticados por particular contra a Administração Pública | Corrupção ativa (Art. 333) | Masculino':'Qtde_Corrupcao_Ativa_M',

                        '5.13.b.7.                                                             Grupo: Crimes praticados por particular contra a Administração Pública | Corrupção ativa (Art. 333) | Feminino':'Qtde_Corrupcao_Ativa_F',

                        '5.13.b.7.                                                             Grupo: Crimes praticados por particular contra a Administração Pública | Contrabando ou descaminho (Art. 334) | Masculino':'Qtde_Contrabando_M',

                        '5.13.b.7.                                                             Grupo: Crimes praticados por particular contra a Administração Pública | Contrabando ou descaminho (Art. 334) | Feminino':'Qtde_Contrabando_F',

                        '5.13.b.8.                                                                       Grupo: Drogas (Lei 6.368/76 e Lei 11.343/06) | Tráfico de drogas (Art. 12 da Lei 6.368/76 e Art. 33 da Lei 11.343/06) | Masculino':'Qtde_Trafico_Drogas_M',

                        '5.13.b.8.                                                                       Grupo: Drogas (Lei 6.368/76 e Lei 11.343/06) | Tráfico de drogas (Art. 12 da Lei 6.368/76 e Art. 33 da Lei 11.343/06) | Feminino':'Qtde_Trafico_Drogas_F',

                        '5.13.b.8.                                                                       Grupo: Drogas (Lei 6.368/76 e Lei 11.343/06) | Associação para o tráfico (Art. 14 da Lei 6.368/76 e Art. 35 da Lei 11.343/06) | Masculino':'Qtde_Associacao_Trafico_M',

                        '5.13.b.8.                                                                       Grupo: Drogas (Lei 6.368/76 e Lei 11.343/06) | Associação para o tráfico (Art. 14 da Lei 6.368/76 e Art. 35 da Lei 11.343/06) | Feminino':'Qtde_Associacao_Trafico_F',

                        '5.13.b.8.                                                                       Grupo: Drogas (Lei 6.368/76 e Lei 11.343/06) | Tráfico internacional de drogas (Art. 18 da Lei 6.368/76 e Art. 33 e 40, inciso I da Lei 11.343/06) | Masculino':'Qtde_Trafico_Intern_Drogas_M',

                        '5.13.b.8.                                                                       Grupo: Drogas (Lei 6.368/76 e Lei 11.343/06) | Tráfico internacional de drogas (Art. 18 da Lei 6.368/76 e Art. 33 e 40, inciso I da Lei 11.343/06) | Feminino':'Qtde_Trafico_Intern_Drogas_F',

                        '5.13.b.9.                                                                     Grupo: Estatuto do Desarmamento (Lei 10.826, de 22/12/2003  | Porte ilegal de arma de fogo de uso permitido (Art. 14) | Masculino':'Qtde_Porte_Ilegal_Armas_Permitida_M',

                        '5.13.b.9.                                                                     Grupo: Estatuto do Desarmamento (Lei 10.826, de 22/12/2003  | Porte ilegal de arma de fogo de uso permitido (Art. 14) | Feminino':'Qtde_Porte_Ilegal_Armas_Permitida_F',

                        '5.13.b.9.                                                                     Grupo: Estatuto do Desarmamento (Lei 10.826, de 22/12/2003  | Disparo de arma de fogo (Art. 15) | Masculino':'Qtde_Disparo_Arma_Fogo_M',

                        '5.13.b.9.                                                                     Grupo: Estatuto do Desarmamento (Lei 10.826, de 22/12/2003  | Disparo de arma de fogo (Art. 15) | Feminino':'Qtde_Disparo_Arma_Fogo_F',

                        '5.13.b.9.                                                                     Grupo: Estatuto do Desarmamento (Lei 10.826, de 22/12/2003  | Posse ou porte ilegal de arma de fogo de uso restrito (Art. 16) | Masculino':'Qtde_Porte_Ilegal_Armas_Restrita_M',

                        '5.13.b.9.                                                                     Grupo: Estatuto do Desarmamento (Lei 10.826, de 22/12/2003  | Posse ou porte ilegal de arma de fogo de uso restrito (Art. 16) | Feminino':'Qtde_Porte_Ilegal_Armas_Restrita_F',

                        '5.13.b.9.                                                                     Grupo: Estatuto do Desarmamento (Lei 10.826, de 22/12/2003  | Comércio ilegal de arma de fogo (Art. 17) | Masculino':'Qtde_Comercio_Ilegal_Armas_M',

                        '5.13.b.9.                                                                     Grupo: Estatuto do Desarmamento (Lei 10.826, de 22/12/2003  | Comércio ilegal de arma de fogo (Art. 17) | Feminino':'Qtde_Comercio_Ilegal_Armas_F',

                        '5.13.b.9.                                                                     Grupo: Estatuto do Desarmamento (Lei 10.826, de 22/12/2003  | Tráfico internacional de arma de fogo (Art. 18) | Masculino':'Qtde_Trafico_Internacional_Armas_M',

                        '5.13.b.9.                                                                     Grupo: Estatuto do Desarmamento (Lei 10.826, de 22/12/2003  | Tráfico internacional de arma de fogo (Art. 18) | Feminino':'Qtde_Trafico_Internacional_Armas_F',

                        '5.13.b.10.                                                                     Grupo: Crimes de Trânsito (Lei 9.503, de 23/09/1997) | Homicídio culposo na condução de veículo automotor (Art. 302) | Masculino':'Qtde_Homic_Culposo_Condu_Veiculo_Automotor_M',

                        '5.13.b.10.                                                                     Grupo: Crimes de Trânsito (Lei 9.503, de 23/09/1997) | Homicídio culposo na condução de veículo automotor (Art. 302) | Feminino':'Qtde_Homic_Culposo_Condu_Veiculo_Automotor_F',

                        '5.13.b.10.                                                                     Grupo: Crimes de Trânsito (Lei 9.503, de 23/09/1997) | Outros (Art. 303 a 312) | Masculino':'Qtde_Cirmes_Transito_Outros_M',

                        '5.13.b.10.                                                                     Grupo: Crimes de Trânsito (Lei 9.503, de 23/09/1997) | Outros (Art. 303 a 312) | Feminino':'Qtde_Cirmes_Transito_Outros_F',

                        '5.13.b.11.                                                                     Grupo: Legislação específica - outros | Estatuto da Criança e do Adolescente (Lei 8.069, de 13/01/1990) | Masculino':'Qtde_Estatuto_Crianca_Adolesc_M',

                        '5.13.b.11.                                                                     Grupo: Legislação específica - outros | Estatuto da Criança e do Adolescente (Lei 8.069, de 13/01/1990) | Feminino':'Qtde_Estatuto_Crianca_Adolesc_F',

                        '5.13.b.11.                                                                     Grupo: Legislação específica - outros | Genocídio (Lei 2.889, de 01/10/1956) | Masculino':'Qtde_Genocidio_M',

                        '5.13.b.11.                                                                     Grupo: Legislação específica - outros | Genocídio (Lei 2.889, de 01/10/1956) | Feminino':'Qtde_Genocidio_F',

                        '5.13.b.11.                                                                     Grupo: Legislação específica - outros | Crimes de tortura (Lei 9.455, de 07/04/1997) | Masculino':'Qtde_Crimes_Tortura_M',

                        '5.13.b.11.                                                                     Grupo: Legislação específica - outros | Crimes de tortura (Lei 9.455, de 07/04/1997) | Feminino':'Qtde_Crimes_Tortura_F',

                        '5.13.b.11.                                                                     Grupo: Legislação específica - outros | Crimes contra o Meio Ambiente (Lei 9.605, de 12/02/1998) | Masculino':'Qtde_Crimes_Contra_Meio_Amb_M',

                        '5.13.b.11.                                                                     Grupo: Legislação específica - outros | Crimes contra o Meio Ambiente (Lei 9.605, de 12/02/1998) | Feminino':'Qtde_Crimes_Contra_Meio_Amb_F',

                        '5.13.c. Total de pessoas privadas de liberdade: | Número de pessoas privadas de liberdade COM informação sobre tipificação criminal | Masculino':'Qtde_Tipificacao_Criminal_M',

                        '5.13.c. Total de pessoas privadas de liberdade: | Número de pessoas privadas de liberdade COM informação sobre tipificação criminal | Feminino':'Qtde_Tipificacao_Criminal_F',

                        '5.13.c. Total de pessoas privadas de liberdade: | Número de pessoas privadas de liberdade SEM informação sobre tipificação criminal | Masculino':'Qtde_Sem_Tipificacao_Criminal_M',

                        '5.13.c. Total de pessoas privadas de liberdade: | Número de pessoas privadas de liberdade SEM informação sobre tipificação criminal | Feminino':'Qtde_Sem_Tipificacao_Criminal_F',

                        '6.1. Existem pessoas privadas de liberdade neste estabelecimento em atividades laborterápicas?':'Tem_Controle_Atividades_Laborterapicas',

                        '6.1.a. Quantidade de pessoas em vagas obtidas por meios próprios e/ou sem intervenção do sistema prisional: | Setor primário - rural, agrícola e artesanato | Trabalho externo Masculino':'Qtde_Trab_Externo_Setor_Primario_M',

                        '6.1.a. Quantidade de pessoas em vagas obtidas por meios próprios e/ou sem intervenção do sistema prisional: | Setor primário - rural, agrícola e artesanato | Trabalho externo Feminino':'Qtde_Trab_Externo_Setor_Primario_F',

                        '6.1.a. Quantidade de pessoas em vagas obtidas por meios próprios e/ou sem intervenção do sistema prisional: | Setor primário - rural, agrícola e artesanato | Trabalho interno Masculino':'Qtde_Trab_Interno_Setor_Primario_M',

                        '6.1.a. Quantidade de pessoas em vagas obtidas por meios próprios e/ou sem intervenção do sistema prisional: | Setor primário - rural, agrícola e artesanato | Trabalho interno Feminino':'Qtde_Trab_Interno_Setor_Primario_F',

                        '6.1.a. Quantidade de pessoas em vagas obtidas por meios próprios e/ou sem intervenção do sistema prisional: | Setor secundário - industrial e construção civil | Trabalho externo Masculino':'Qtde_Trab_Externo_Setor_Secundario_M',

                        '6.1.a. Quantidade de pessoas em vagas obtidas por meios próprios e/ou sem intervenção do sistema prisional: | Setor secundário - industrial e construção civil | Trabalho externo Feminino':'Qtde_Trab_Externo_Setor_Secundario_F',

                        '6.1.a. Quantidade de pessoas em vagas obtidas por meios próprios e/ou sem intervenção do sistema prisional: | Setor secundário - industrial e construção civil | Trabalho interno Masculino':'Qtde_Trab_Interno_Setor_Secundario_M',

                        '6.1.a. Quantidade de pessoas em vagas obtidas por meios próprios e/ou sem intervenção do sistema prisional: | Setor secundário - industrial e construção civil | Trabalho interno Feminino':'Qtde_Trab_Interno_Setor_Secundario_F',

                        '6.1.a. Quantidade de pessoas em vagas obtidas por meios próprios e/ou sem intervenção do sistema prisional: | Setor terciário - serviços | Trabalho externo Masculino':'Qtde_Trab_Externo_Setor_Terceario_M',

                        '6.1.a. Quantidade de pessoas em vagas obtidas por meios próprios e/ou sem intervenção do sistema prisional: | Setor terciário - serviços | Trabalho externo Feminino':'Qtde_Trab_Externo_Setor_Terceario_F',

                        '6.1.a. Quantidade de pessoas em vagas obtidas por meios próprios e/ou sem intervenção do sistema prisional: | Setor terciário - serviços | Trabalho interno Masculino':'Qtde_Trab_Interno_Setor_Terceario_M',

                        '6.1.a. Quantidade de pessoas em vagas obtidas por meios próprios e/ou sem intervenção do sistema prisional: | Setor terciário - serviços | Trabalho interno Feminino':'Qtde_Trab_Interno_Setor_Terceario_F',

                        '6.1.b. Quantidade de pessoas em vagas disponibilizadas pela administração prisional como apoio ao próprio estabelecimento: | Apoio ao estabelecimento | Trabalho interno Masculino':'Qtde_Trab_Apoio_Estb_Interno_M',

                        '6.1.b. Quantidade de pessoas em vagas disponibilizadas pela administração prisional como apoio ao próprio estabelecimento: | Apoio ao estabelecimento | Trabalho interno Feminino':'Qtde_Trab_Apoio_Estb_Interno_F',

                        '6.1.c. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com a iniciativa privada: | Setor primário - rural, agrícola e artesanato | Trabalho externo Masculino':'Qtde_Trab_Externo_Parc_Privada_Setor_Primario_M',

                        '6.1.c. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com a iniciativa privada: | Setor primário - rural, agrícola e artesanato | Trabalho externo Feminino':'Qtde_Trab_Externo_Parc_Privada_Setor_Primario_F',

                        '6.1.c. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com a iniciativa privada: | Setor primário - rural, agrícola e artesanato | Trabalho interno Masculino':'Qtde_Trab_Interno_Parc_Privada_Setor_Primario_M',

                        '6.1.c. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com a iniciativa privada: | Setor primário - rural, agrícola e artesanato | Trabalho interno Feminino':'Qtde_Trab_Interno_Parc_Privada_Setor_Primario_F',

                        '6.1.c. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com a iniciativa privada: | Setor secundário - industrial e construção civil | Trabalho externo Masculino':'Qtde_Trab_Externo_Parc_Privada_Setor_Secundario_M',

                        '6.1.c. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com a iniciativa privada: | Setor secundário - industrial e construção civil | Trabalho externo Feminino':'Qtde_Trab_Externo_Parc_Privada_Setor_Secundario_F',

                        '6.1.c. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com a iniciativa privada: | Setor secundário - industrial e construção civil | Trabalho interno Masculino':'Qtde_Trab_Interno_Parc_Privada_Setor_Secundario_M',

                        '6.1.c. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com a iniciativa privada: | Setor secundário - industrial e construção civil | Trabalho interno Feminino':'Qtde_Trab_Interno_Parc_Privada_Setor_Secundario_F',

                        '6.1.c. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com a iniciativa privada: | Setor terciário - serviços | Trabalho externo Masculino':'Qtde_Trab_Externo_Parc_Privada_Setor_Terceario_M',

                        '6.1.c. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com a iniciativa privada: | Setor terciário - serviços | Trabalho externo Feminino':'Qtde_Trab_Externo_Parc_Privada_Setor_Terceario_F',

                        '6.1.c. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com a iniciativa privada: | Setor terciário - serviços | Trabalho interno Masculino':'Qtde_Trab_Interno_Parc_Privada_Setor_Terceario_M',

                        '6.1.c. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com a iniciativa privada: | Setor terciário - serviços | Trabalho interno Feminino':'Qtde_Trab_Interno_Parc_Privada_Setor_Terceario_F',

                        '6.1.d. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com outros órgãos públicos: | Setor primário - rural, agrícola e artesanato | Trabalho externo Masculino':'Qtde_Trab_Externo_Parc_Publico_Setor_Primario_M',

                        '6.1.d. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com outros órgãos públicos: | Setor primário - rural, agrícola e artesanato | Trabalho externo Feminino':'Qtde_Trab_Externo_Parc_Publico_Setor_Primario_F',

                        '6.1.d. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com outros órgãos públicos: | Setor primário - rural, agrícola e artesanato | Trabalho interno Masculino':'Qtde_Trab_Interno_Parc_Publico_Setor_Primario_M',

                        '6.1.d. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com outros órgãos públicos: | Setor primário - rural, agrícola e artesanato | Trabalho interno Feminino':'Qtde_Trab_Interno_Parc_Publico_Setor_Primario_F',

                        '6.1.d. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com outros órgãos públicos: | Setor secundário - industrial e construção civil | Trabalho externo Masculino':'Qtde_Trab_Externo_Parc_Publico_Setor_Secundario_M',

                        '6.1.d. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com outros órgãos públicos: | Setor secundário - industrial e construção civil | Trabalho externo Feminino':'Qtde_Trab_Externo_Parc_Publico_Setor_Secundario_F',

                        '6.1.d. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com outros órgãos públicos: | Setor secundário - industrial e construção civil | Trabalho interno Masculino':'Qtde_Trab_Interno_Parc_Publico_Setor_Secundario_M',

                        '6.1.d. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com outros órgãos públicos: | Setor secundário - industrial e construção civil | Trabalho interno Feminino':'Qtde_Trab_Interno_Parc_Publico_Setor_Secundario_F',

                        '6.1.d. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com outros órgãos públicos: | Setor terciário - serviços | Trabalho externo Masculino':'Qtde_Trab_Externo_Parc_Publico_Setor_Terceario_M',

                        '6.1.d. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com outros órgãos públicos: | Setor terciário - serviços | Trabalho externo Feminino':'Qtde_Trab_Externo_Parc_Publico_Setor_Terceario_F',

                        '6.1.d. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com outros órgãos públicos: | Setor terciário - serviços | Trabalho interno Masculino':'Qtde_Trab_Interno_Parc_Publico_Setor_Terceario_M',

                        '6.1.d. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com outros órgãos públicos: | Setor terciário - serviços | Trabalho interno Feminino':'Qtde_Trab_Interno_Parc_Publico_Setor_Terceario_F',

                        '6.1.e. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com entidade ou organizações não governamentais sem fins lucrativos: | Setor primário - rural, agrícola e artesanato | Trabalho externo Masculino':'Qtde_Trab_Externo_Sem_Fim_Luc_Setor_Primario_M',

                        '6.1.e. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com entidade ou organizações não governamentais sem fins lucrativos: | Setor primário - rural, agrícola e artesanato | Trabalho externo Feminino':'Qtde_Trab_Externo_Sem_Fim_Luc_Setor_Primario_F',

                        '6.1.e. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com entidade ou organizações não governamentais sem fins lucrativos: | Setor primário - rural, agrícola e artesanato | Trabalho interno Masculino':'Qtde_Trab_Interno_Sem_Fim_Luc_Setor_Primario_M',

                        '6.1.e. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com entidade ou organizações não governamentais sem fins lucrativos: | Setor primário - rural, agrícola e artesanato | Trabalho interno Feminino':'Qtde_Trab_Interno_Sem_Fim_Luc_Setor_Primario_F',

                        '6.1.e. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com entidade ou organizações não governamentais sem fins lucrativos: | Setor secundário - industrial e construção civil | Trabalho externo Masculino':'Qtde_Trab_Externo_Sem_Fim_Luc_Setor_Secundario_M',

                        '6.1.e. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com entidade ou organizações não governamentais sem fins lucrativos: | Setor secundário - industrial e construção civil | Trabalho externo Feminino':'Qtde_Trab_Externo_Sem_Fim_Luc_Setor_Secundario_F',

                        '6.1.e. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com entidade ou organizações não governamentais sem fins lucrativos: | Setor secundário - industrial e construção civil | Trabalho interno Masculino':'Qtde_Trab_Interno_Sem_Fim_Luc_Setor_Secundario_',

                        '6.1.e. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com entidade ou organizações não governamentais sem fins lucrativos: | Setor secundário - industrial e construção civil | Trabalho interno Feminino':'Qtde_Trab_Interno_Sem_Fim_Luc_Setor_Secundario_F',

                        '6.1.e. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com entidade ou organizações não governamentais sem fins lucrativos: | Setor terciário - serviços | Trabalho externo Masculino':'Qtde_Trab_Externo_Sem_Fim_Luc_Setor_Terceario_M',

                        '6.1.e. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com entidade ou organizações não governamentais sem fins lucrativos: | Setor terciário - serviços | Trabalho externo Feminino':'Qtde_Trab_Externo_Sem_Fim_Luc_Setor_Terceario_F',

                        '6.1.e. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com entidade ou organizações não governamentais sem fins lucrativos: | Setor terciário - serviços | Trabalho interno Masculino':'Qtde_Trab_Interno_Sem_Fim_Luc_Setor_Terceario_M',

                        '6.1.e. Quantidade de pessoas em vagas disponibilizadas pela administração prisional em parceria com entidade ou organizações não governamentais sem fins lucrativos: | Setor terciário - serviços | Trabalho interno Feminino':'Qtde_Trab_Interno_Sem_Fim_Luc_Setor_Terceario_F',

                        '6.2. Quantidade de pessoas privadas de liberdade por remuneraçãoIdentificar o valor médio percebido pelas pessoas privadas de liberdade em razão do trabalho em 30/06/2014, de acordo com os registros do estabelecimento. O estabelecimento tem condições de obter estas informações em seus registros?':'Tem_Controle_Remuneracao',

                        '6.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Não recebe | Masculino':'Qtde_Nao_Tem_Remuneracao_M',

                        '6.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Não recebe | Feminino':'Qtde_Nao_Tem_Remuneracao_F',

                        '6.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Menos do que 3/4 do salário mínimo mensal | Masculino':'Qtde_Remuneracao_3_4_Sal_Min_M',

                        '6.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Menos do que 3/4 do salário mínimo mensal | Feminino':'Qtde_Remuneracao_3_4_Sal_Min_F',

                        '6.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Entre 3/4 e 1 salário mínimo mensal | Masculino':'Qtde_Remuneracao_3_4_Mais_1_Sal_Min_M',

                        '6.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Entre 3/4 e 1 salário mínimo mensal | Feminino':'Qtde_Remuneracao_3_4_Mais_1_Sal_Min_F',

                        '6.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Entre 1 e 2 salários mínimos mensais | Masculino':'Qtde_Remuneracao_1_2_Sal_Min_M',

                        '6.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Entre 1 e 2 salários mínimos mensais | Feminino':'Qtde_Remuneracao_1_2_Sal_Min_F',

                        '6.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais que 2 salários mínimos mensais | Masculino':'Qtde_Remuneracao_Mais_2_Sal_Min_M',

                        '6.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Mais que 2 salários mínimos mensais | Feminino':'Qtde_Remuneracao_Mais_2_Sal_Min_F',

                        '6.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Sem informação | Masculino':'Qtde_Remuneracao_Sem_Inf_M',

                        '6.2.a. Em caso positivo, total ou parcialmente, preencha as informações abaixo: | Sem informação | Feminino':'Qtde_Remuneracao_Sem_Inf_F',

                        '6.3. Existem pessoas privadas de liberdade neste estabelecimento em atividades educacionais?':'Tem_Controle_Atividades_Educacionais',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Total de pessoas em atividade educacional | Presencial Masculino':'Qtde_Presencial_Atividades_Educacionais_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Total de pessoas em atividade educacional | Presencial Feminino':'Qtde_Presencial_Atividades_Educacionais_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Total de pessoas em atividade educacional | Ensino à distância Masculino':'Qtde_EAD_Atividades_Educacionais_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Total de pessoas em atividade educacional | Ensino à distância Feminino':'Qtde_EAD_Atividades_Educacionais_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Total de pessoas em atividade educacional | Certificação/ conclusão no semestre Masculino':'Qtde_Certif_Atividade_Educacional_Semestre_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Total de pessoas em atividade educacional | Certificação/ conclusão no semestre Feminino':'Qtde_Certif_Atividade_Educacional_Semestre_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Alfabetização | Presencial Masculino':'Qtde_Presencial_Alfabetizacao_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Alfabetização | Presencial Feminino':'Qtde_Presencial_Alfabetizacao_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Alfabetização | Ensino à distância Masculino':'Qtde_EAD_Alfabetizacao_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Alfabetização | Ensino à distância Feminino':'Qtde_EAD_Alfabetizacao_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Alfabetização | Certificação/ conclusão no semestre Masculino':'Qtde_Certif_Alfabetizacao_Semestre_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Alfabetização | Certificação/ conclusão no semestre Feminino':'Qtde_Certif_Alfabetizacao_Semestre_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Fundamental | Presencial Masculino':'Qtde_Presencial_Ensino_Fundamental_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Fundamental | Presencial Feminino':'Qtde_Presencial_Ensino_Fundamental_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Fundamental | Ensino à distância Masculino':'Qtde_EAD_Ensino_Fundamental_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Fundamental | Ensino à distância Feminino':'Qtde_EAD_Ensino_Fundamental_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Fundamental | Certificação/ conclusão no semestre Masculino':'Qtde_Certif_Ensino_Fundamental_Semestre_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Fundamental | Certificação/ conclusão no semestre Feminino':'Qtde_Certif_Ensino_Fundamental_Semestre_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Médio | Presencial Masculino':'Qtde_Presencial_Ensino_Medio_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Médio | Presencial Feminino':'Qtde_Presencial_Ensino_Medio_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Médio | Ensino à distância Masculino':'Qtde_EAD_Ensino_Medio_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Médio | Ensino à distância Feminino':'Qtde_EAD_Ensino_Medio_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Médio | Certificação/ conclusão no semestre Masculino':'Qtde_Certif_Ensino_Medio_Semestre_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Médio | Certificação/ conclusão no semestre Feminino':'Qtde_Certif_Ensino_Medio_Semestre_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Superior | Presencial Masculino':'Qtde_Presencial_Ensino_Superior_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Superior | Presencial Feminino':'Qtde_Presencial_Ensino_Superior_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Superior | Ensino à distância Masculino':'Qtde_EAD_Ensino_Superior_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Superior | Ensino à distância Feminino':'Qtde_EAD_Ensino_Superior_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Superior | Certificação/ conclusão no semestre Masculino':'Qtde_Certif_Ensino_Superior_Semestre_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Ensino Superior | Certificação/ conclusão no semestre Feminino':'Qtde_Certif_Ensino_Superior_Semestre_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Curso Técnico (acima de 800 horas de aula) | Presencial Masculino':'Qtde_Presencial_Curso_Tec_800h_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Curso Técnico (acima de 800 horas de aula) | Presencial Feminino':'Qtde_Presencial_Curso_Tec_800h_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Curso Técnico (acima de 800 horas de aula) | Ensino à distância Masculino':'Qtde_EAD_Curso_Tec_800h_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Curso Técnico (acima de 800 horas de aula) | Ensino à distância Feminino':'Qtde_EAD_Curso_Tec_800h_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Curso Técnico (acima de 800 horas de aula) | Certificação/ conclusão no semestre Masculino':'Qtde_Certif_Semestre_Curso_Tec_800h_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Curso Técnico (acima de 800 horas de aula) | Certificação/ conclusão no semestre Feminino':'Qtde_Certif_Semestre_Curso_Tec_800h_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Curso de Formação Inicial e Continuada (Capacitação Profissional, acima de 160 horas de aula) | Presencial Masculino':'Qtde_Presencial_Capacitacao_Profissional_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Curso de Formação Inicial e Continuada (Capacitação Profissional, acima de 160 horas de aula) | Presencial Feminino':'Qtde_Presencial_Capacitacao_Profissional_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Curso de Formação Inicial e Continuada (Capacitação Profissional, acima de 160 horas de aula) | Ensino à distância Masculino':'Qtde_EAD_Capacitacao_Profissional_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Curso de Formação Inicial e Continuada (Capacitação Profissional, acima de 160 horas de aula) | Ensino à distância Feminino':'Qtde_EAD_Capacitacao_Profissional_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Curso de Formação Inicial e Continuada (Capacitação Profissional, acima de 160 horas de aula) | Certificação/ conclusão no semestre Masculino':'Qtde_Certif_Capacitacao_Profissional_Semestre_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Curso de Formação Inicial e Continuada (Capacitação Profissional, acima de 160 horas de aula) | Certificação/ conclusão no semestre Feminino':'Qtde_Certif_Capacitacao_Profissional_Semestre_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas matriculadas em programa de remição pelo estudo através da leitura | Presencial Masculino':'Qtde_Presencial_Remicao_Leitura_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas matriculadas em programa de remição pelo estudo através da leitura | Presencial Feminino':'Qtde_Presencial_Remicao_Leitura_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas matriculadas em programa de remição pelo estudo através da leitura | Ensino à distância Masculino':'Qtde_EAD_Remicao_Leitura_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas matriculadas em programa de remição pelo estudo através da leitura | Ensino à distância Feminino':'Qtde_EAD_Remicao_Leitura_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas matriculadas em programa de remição pelo estudo através da leitura | Certificação/ conclusão no semestre Masculino':'Qtde_Certif_Remicao_Leitura_Semestre_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas matriculadas em programa de remição pelo estudo através da leitura | Certificação/ conclusão no semestre Feminino':'Qtde_Certif_Remicao_Leitura_Semestre_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas matriculadas em programa de remição pelo estudo através do esporte | Presencial Masculino':'Qtde_Presencial_Remicao_Esporte_Semestre_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas matriculadas em programa de remição pelo estudo através do esporte | Presencial Feminino':'Qtde_Presencial_Remicao_Esporte_Semestre_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas matriculadas em programa de remição pelo estudo através do esporte | Ensino à distância Masculino':'Qtde_EAD_Remicao_Esporte_Semestre_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas matriculadas em programa de remição pelo estudo através do esporte | Ensino à distância Feminino':'Qtde_EAD_Remicao_Esporte_Semestre_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas matriculadas em programa de remição pelo estudo através do esporte | Certificação/ conclusão no semestre Masculino':'Qtde_Certif_Remicao_Esporte_Semestre_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas matriculadas em programa de remição pelo estudo através do esporte | Certificação/ conclusão no semestre Feminino':'Qtde_Certif_Remicao_Esporte_Semestre_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas envolvidas em atividades educacionais complementares (videoteca, atividades de lazer, cultura) | Presencial Masculino':'Qtde_Presencial_Remicao_Atividades_Complem_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas envolvidas em atividades educacionais complementares (videoteca, atividades de lazer, cultura) | Presencial Feminino':'Qtde_Presencial_Remicao_Atividades_Complem_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas envolvidas em atividades educacionais complementares (videoteca, atividades de lazer, cultura) | Ensino à distância Masculino':'Qtde_EAD_Remicao_Atividades_Complem_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas envolvidas em atividades educacionais complementares (videoteca, atividades de lazer, cultura) | Ensino à distância Feminino':'Qtde_EAD_Remicao_Atividades_Complem_F',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas envolvidas em atividades educacionais complementares (videoteca, atividades de lazer, cultura) | Certificação/ conclusão no semestre Masculino':'Qtde_Certif_Atividades_Complem_Semestre_M',

                        '6.3.a. Quantidade de pessoas privadas de liberdade em atividade educacional: | Pessoas envolvidas em atividades educacionais complementares (videoteca, atividades de lazer, cultura) | Certificação/ conclusão no semestre Feminino':'Qtde_Certif_Atividades_Complem_Semestre_F',

                        '6.4. Quantidade de pessoas trabalhando e estudando, simultaneamente: | Pessoas que trabalham e estudam | Masculino':'Qtde_Trabalham_E_Estudam_M',

                        '6.4. Quantidade de pessoas trabalhando e estudando, simultaneamente: | Pessoas que trabalham e estudam | Feminino':'Qtde_Trabalham_E_Estudam_F',

                        '6.5. Quantidade de famílias que recebem auxílio-reclusãoNúmero de famílias que recebem auxílio-reclusão em 30/06/2014. O estabelecimento detém alguma forma de registro que permite a obtenção desta informação?':'Tem_Controle_Familias_Receb_Aux_Exclusao',

                        '6.5. Número de pessoas privadas de liberdade cujas famílias recebem auxílio-reclusão: | Número de pessoas privadas de liberdade cujas famílias recebem auxílio-reclusão | Masculino':'Qtde_Familias_Receb_Aux_Exclusao_M',

                        '6.5. Número de pessoas privadas de liberdade cujas famílias recebem auxílio-reclusão: | Número de pessoas privadas de liberdade cujas famílias recebem auxílio-reclusão | Feminino':'Qtde_Familias_Receb_Aux_Exclusao_F',

                        '6.6. Informações da área de saúde - total do semestre: | Consultas médicas realizadas externamente | Masculino':'Qtde_Consult_Medicas_Externas_M',

                        '6.6. Informações da área de saúde - total do semestre: | Consultas médicas realizadas externamente | Feminino':'Qtde_Consult_Medicas_Externas_F',

                        '6.6. Informações da área de saúde - total do semestre: | Consultas médicas realizadas no estabelecimento | Masculino':'Qtde_Consult_Medicas_Internas_M',

                        '6.6. Informações da área de saúde - total do semestre: | Consultas médicas realizadas no estabelecimento | Feminino':'Qtde_Consult_Medicas_Internas_F',

                        '6.6. Informações da área de saúde - total do semestre: | Consultas psicológicas | Masculino':'Qtde_Consult_Piscologicas_M',

                        '6.6. Informações da área de saúde - total do semestre: | Consultas psicológicas | Feminino':'Qtde_Consult_Piscologicas_F',

                        '6.6. Informações da área de saúde - total do semestre: | Consultas odontológicas | Masculino':'Qtde_Consult_Odontologicas_M',

                        '6.6. Informações da área de saúde - total do semestre: | Consultas odontológicas | Feminino':'Qtde_Consult_Odontologicas_F',

                        '6.6. Informações da área de saúde - total do semestre: | Quantidade de exames e testagem | Masculino':'Qtde_Exames_M',

                        '6.6. Informações da área de saúde - total do semestre: | Quantidade de exames e testagem | Feminino':'Qtde_Exames_F',

                        '6.6. Informações da área de saúde - total do semestre: | Quantidade de intervenções cirúrgicas | Masculino':'Qtde_Inter_Cirurgicas_M',

                        '6.6. Informações da área de saúde - total do semestre: | Quantidade de intervenções cirúrgicas | Feminino':'Qtde_Inter_Cirurgicas_F',

                        '6.6. Informações da área de saúde - total do semestre: | Quantidade de vacinas | Masculino':'Qtde_Vacinas_M',

                        '6.6. Informações da área de saúde - total do semestre: | Quantidade de vacinas | Feminino':'Qtde_Vacinas_F',

                        '6.6. Informações da área de saúde - total do semestre: | Quantidade de outros procedimentos, como sutura e curativo | Masculino':'Qtde_Outros_Proc_Saude_M',

                        '6.6. Informações da área de saúde - total do semestre: | Quantidade de outros procedimentos, como sutura e curativo | Feminino':'Qtde_Outros_Proc_Saude_F',

                        '6.7. Quantidade de pessoas com agravos transmissíveis em 30/06/2014: | HIV | Masculino':'Qtde_HIV_Positivo_M',

                        '6.7. Quantidade de pessoas com agravos transmissíveis em 30/06/2014: | HIV | Feminino':'Qtde_HIV_Positivo_F',

                        '6.7. Quantidade de pessoas com agravos transmissíveis em 30/06/2014: | Sífilis | Masculino':'Qtde_Sifilis_Positivo_M',

                        '6.7. Quantidade de pessoas com agravos transmissíveis em 30/06/2014: | Sífilis | Feminino':'Qtde_Sifilis_Positivo_F',

                        '6.7. Quantidade de pessoas com agravos transmissíveis em 30/06/2014: | Hepatite | Masculino':'Qtde_Hepatite_Positivo_M',

                        '6.7. Quantidade de pessoas com agravos transmissíveis em 30/06/2014: | Hepatite | Feminino':'Qtde_Hepatite_Positivo_F',

                        '6.7. Quantidade de pessoas com agravos transmissíveis em 30/06/2014: | Tuberculose | Masculino':'Qtde_Tuberculose_Positivo_M',

                        '6.7. Quantidade de pessoas com agravos transmissíveis em 30/06/2014: | Tuberculose | Feminino':'Qtde_Tuberculose_Positivo_F',

                        '6.7. Quantidade de pessoas com agravos transmissíveis em 30/06/2014: | Outros. Quais? | Masculino':'Qtde_DST_Outros_M',

                        '6.7. Quantidade de pessoas com agravos transmissíveis em 30/06/2014: | Outros. Quais? | Feminino':'Qtde_DST_Outros_F',

                        '6.8. Mortalidade no Sistema Prisional (total do semestre): | Óbitos naturais/ óbitos por motivos de saúde | Masculino':'Qtde_Obitos_Naturais_M',

                        '6.8. Mortalidade no Sistema Prisional (total do semestre): | Óbitos naturais/ óbitos por motivos de saúde | Feminino':'Qtde_Obitos_Naturais_F',

                        '6.8. Mortalidade no Sistema Prisional (total do semestre): | Óbitos criminais | Masculino':'Qtde_Obitos_Criminais_M',

                        '6.8. Mortalidade no Sistema Prisional (total do semestre): | Óbitos criminais | Feminino':'Qtde_Obitos_Criminais_F',

                        '6.8. Mortalidade no Sistema Prisional (total do semestre): | Óbitos suicídios | Masculino':'Qtde_Obitos_Suicidios_M',

                        '6.8. Mortalidade no Sistema Prisional (total do semestre): | Óbitos suicídios | Feminino':'Qtde_Obitos_Suicidios_F',

                        '6.8. Mortalidade no Sistema Prisional (total do semestre): | Óbitos acidentais | Masculino':'Qtde_Obitos_Acidentais_M',

                        '6.8. Mortalidade no Sistema Prisional (total do semestre): | Óbitos acidentais | Feminino':'Qtde_Obitos_Acidentais_F',

                        '6.8. Mortalidade no Sistema Prisional (total do semestre): | Óbitos com causa desconhecida | Masculino':'Qtde_Obitos_Causa_Desconhecidas_M',

                        '6.8. Mortalidade no Sistema Prisional (total do semestre): | Óbitos com causa desconhecida | Feminino':'Qtde_Obitos_Causa_Desconhecidas_F',

                        '7.3. Quantidade de visitas registradas no semestreDevem ser computadas todas as visitas registradas entre 01/01/2014 e 30/06/2014. Vale ressaltar que se uma mesma pessoa realizou várias visitas, devem ser computadas todas as visitas realizadas por esta pessoa. O estabelecimento detém alguma forma de registro que permite a obtenção desta informação?':'Tem_Controle_Visitas_Computadas',

                        '7.3.a. Quantidade de visitas registradas no semestre:':'Qtde_Visitas_Registradas_Semestre',

                        '7.3.b. Quantidade de presos que receberam visita no semestre: | Quantidade de presos que receberam visita no semestre | Masculino':'Qtde_Receberam_Visitas_Semestre_M',

                        '7.3.b. Quantidade de presos que receberam visita no semestre: | Quantidade de presos que receberam visita no semestre | Feminino':'Qtde_Receberam_Visitas_Semestre_F',

                        '7.5. Foi realizada visita(s) de inspeção no semestre?':'Inspecao_No_Semestre',

                        '7.5.a. Por qual(is) órgão(s)? [Conselho Nacional de Política Criminal e Penitenciária - CNPCP]':'Inspecao_No_Semestre_CNPCP',

                        '7.5.a. Por qual(is) órgão(s)? [Conselho Estadual de Política Criminal e Penitenciária/ Conselho Penitenciário]':'Inspecao_No_Semestre_Conselho_Penintenciario',

                        '7.5.a. Por qual(is) órgão(s)? [Conselho da Comunidade]':'Inspecao_No_Semestre_Conselho_Comunidade',

                        '7.5.a. Por qual(is) órgão(s)? [Ouvidoria do sistema prisional - estadual ou nacional]':'Inspecao_No_Semestre_Ouvidoria_Sistem_Prisional',

                        '7.5.a. Por qual(is) órgão(s)? [Defensoria Pública]':'Inspecao_No_Semestre_Defensoria_Publica',

                        '7.5.a. Por qual(is) órgão(s)? [Judiciário]':'Inspecao_No_Semestre_Judiciario',

                        '7.5.a. Por qual(is) órgão(s)? [Ministério Público]':'Inspecao_No_Semestre_Ministerio_Publico',

                        '7.5.a. Por qual(is) órgão(s)? [Outro(s). Qual?]':'Inspecao_No_Semestre_Outros',

                        '7.5.a. Por qual(is) órgão(s)? [Outro(s). Qual?] [text]':'Inspecao_No_Semestre_Outros_Desc'              

                       },

               inplace=True)
df_2014.head(1)
df_2014.dtypes
df_2014.isnull().sum()
df_2014.describe().T
df_2014 = df_2014.replace("-", 0)
df_2014 = df_2014.replace("NaN", 0)
df_2014.update(df_2014.fillna(0))
df_2014 = df_2014.replace("Complete", 1)

df_2014 = df_2014.replace("Incomplete", 0)
df_2014['Capacidade_Pop_Prisional'] = df_2014['Capacidade_Prov_M'].astype(int)+ df_2014['Capacidade_Prov_F'].astype(int)+ df_2014['Capacidade_Fechado_M'].astype(int)+ df_2014['Capacidade_Fechado_F'].astype(int)+ df_2014['Capacidade_Semi_F'].astype(int)+ df_2014['Capacidade_Semi_M'].astype(int)+ df_2014['Capacidade_Aberto_M'].astype(int)+ df_2014['Capacidade_Aberto_F'].astype(int)+ df_2014['Capacidade_RDD_M'].astype(int)+ df_2014['Capacidade_RDD_F'].astype(int)+ df_2014['Capacidade_Intern_M'].astype(int)+ df_2014['Capacidade_Intern_F'].astype(int)+ df_2014['Capacidade_Outros_M'].astype(int)+ df_2014['Capacidade_Outros_F'].astype(int)
df_2014.update(df_2014['Qtde_Gestantes'].fillna(0))

df_2014.update(df_2014['Qtde_Lactantes'].fillna(0))
df_2014['Qtde_Gestantes']= df_2014['Qtde_Gestantes'].replace("nãonão", 0)

df_2014['Qtde_Lactantes']= df_2014['Qtde_Lactantes'].replace("nãonão", 0)
df_2014['Qtde_Gestantes']= df_2014['Qtde_Gestantes'].replace("não", 0)

df_2014['Qtde_Lactantes']= df_2014['Qtde_Lactantes'].replace("não", 0)
df_2014['Qtde_Gestantes']= df_2014['Qtde_Gestantes'].replace("NãoNão", 0)

df_2014['Qtde_Lactantes']= df_2014['Qtde_Lactantes'].replace("NãoNão", 0)
df_2014['Qtde_Gestantes']= df_2014['Qtde_Gestantes'].replace("Não", 0)

df_2014['Qtde_Lactantes']= df_2014['Qtde_Lactantes'].replace("Não", 0)
df= df_2014[df_2014['UF'] == 0]

df
df2 = df_2014[df_2014['UF'] == 'Rondonia']

df2
df1 = df_2014[df_2014['UF'] == 'MS']

df1
df_2014.loc[1041,'UF'] = 'Rondônia (RO)'

df_2014.loc[605,'UF'] = 'Mato Grosso do Sul (MS)'

df_2014.loc[606,'UF'] = 'Mato Grosso do Sul (MS)'

df_2014.loc[623,'UF'] = 'Mato Grosso do Sul (MS)'

df_2014.loc[101,'UF'] = 'Ceará (CE)'

df_2014.loc[147,'UF'] = 'Ceará (CE)'

df_2014.loc[238,'UF'] = 'Espírito Santo (ES)'

df_2014.loc[379,'UF'] = 'Maranhão (MA)'

df_2014.loc[890,'UF'] = 'Piauí (PI)'

df_2014.loc[1058,'UF'] = 'Rondônia (RO)'

df_2014.loc[1178,'UF'] = 'Santa Catarina (SC)'
df_2014_UF = df_2014[['Unidade_Prisional','UF']]
ax = df_2014['Status'].value_counts().plot(kind='bar', figsize=(10,7),color="indigo", fontsize=10);

ax.set_alpha(0.8)

ax.set_title("Cadastros Preenchidos", fontsize=18)

ax.set_ylabel("Frequencia", fontsize=18);

ax.set_xticklabels(["Completo", "Incompleto"], rotation=0, fontsize=13)



totals = []

for i in ax.patches:

    totals.append(i.get_height())



total = sum(totals)

for i in ax.patches:

    ax.text(i.get_x()+.12, i.get_height()-3, \

            str(round((i.get_height()/total)*100, 2))+'%', fontsize=22,color='black')
title = "Capacidade Total dos Presidios"

Capacidade_Pop_Prisional = df_2014['Capacidade_Pop_Prisional'].astype(int).sum()

Markdown('<strong>{}</strong><br/>{}'.format(title, Capacidade_Pop_Prisional))
df_2014['Total_Pop_Prisional'] = df_2014['Pop_Prisional_Prov_Just_Est_M'].astype(int)+ df_2014['Pop_Prisional_Prov_Just_Est_F'].astype(int)+ df_2014['Pop_Prisional_Prov_Just_Fed_M'].astype(int)+ df_2014['Pop_Prisional_Prov_Just_Fed_F'].astype(int)+ df_2014['Pop_Prisional_Prov_Outros_M'].astype(int)+ df_2014['Pop_Prisional_Prov_Outros_F'].astype(int)+ df_2014['Pop_Prisional_Fechado_Just_Est_M'].astype(int)+ df_2014['Pop_Prisional_Fechado_Just_Est_F'].astype(int)+ df_2014['Pop_Prisional_Fechado_Just_Fed_M'].astype(int)+ df_2014['Pop_Prisional_Fechado_Just_Fed_F'].astype(int)+ df_2014['Pop_Prisional_Fechado_Outros_M'].astype(int)+ df_2014['Pop_Prisional_Fechado_Outros_F'].astype(int)+ df_2014['Pop_Prisional_Semi_Just_Est_M'].astype(int)+ df_2014['Pop_Prisional_Semi_Just_Est_F'].astype(int)+ df_2014['Pop_Prisional_Semi_Just_Fed_M'].astype(int)+ df_2014['Pop_Prisional_Semi_Just_Fed_F'].astype(int)+ df_2014['Pop_Prisional_Semi_Outros_M'].astype(int)+ df_2014['Pop_Prisional_Semi_Outros_F'].astype(int)+ df_2014['Pop_Prisional_Aberto_Just_Est_M'].astype(int)+ df_2014['Pop_Prisional_Aberto_Just_Est_F'].astype(int)+ df_2014['Pop_Prisional_Aberto_Just_Fed_F'].astype(int)+ df_2014['Pop_Prisional_Aberto_Just_Fed_M'].astype(int)+ df_2014['Pop_Prisional_Aberto_Outros_M'].astype(int)+ df_2014['Pop_Prisional_Aberto_Outros_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Est_Internacao_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Est_Internacao_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Fed_Internacao_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Fed_Internacao_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Outros_Internacao_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Outros_Internacao_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Est_Ambulatorial_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Est_Ambulatorial_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Fed_Ambulatorial_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Fed_Ambulatorial_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Outros_Ambulatorial_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Outros_Ambulatorial_F'].astype(int)+df_2014['Qtde_Regime_Disc_Diferenciado'].astype(int)
title = "Total da População Prisional"

Total_Pop_Prisional = df_2014['Total_Pop_Prisional'].sum()

Markdown('<strong>{}</strong><br/>{}'.format(title, Total_Pop_Prisional))
df_2014['Total_Pop_Prisional_F'] = df_2014['Pop_Prisional_Prov_Just_Est_F'].astype(int)+ df_2014['Pop_Prisional_Prov_Just_Fed_F'].astype(int)+ df_2014['Pop_Prisional_Prov_Outros_F'].astype(int)+ df_2014['Pop_Prisional_Fechado_Just_Fed_F'].astype(int)+ df_2014['Pop_Prisional_Fechado_Just_Fed_F'].astype(int)+ df_2014['Pop_Prisional_Fechado_Outros_F'].astype(int)+ df_2014['Pop_Prisional_Semi_Just_Est_F'].astype(int)+ df_2014['Pop_Prisional_Semi_Just_Fed_F'].astype(int)+ df_2014['Pop_Prisional_Semi_Outros_F'].astype(int)+ df_2014['Pop_Prisional_Aberto_Just_Est_F'].astype(int)+ df_2014['Pop_Prisional_Aberto_Just_Fed_F'].astype(int)+ df_2014['Pop_Prisional_Aberto_Just_Fed_M'].astype(int)+ df_2014['Pop_Prisional_Aberto_Outros_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Est_Internacao_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Fed_Internacao_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Outros_Internacao_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Est_Ambulatorial_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Fed_Ambulatorial_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Outros_Ambulatorial_F'].astype(int)
title = "Total da População Prisional Feminina"

Total_Pop_Prisional_F = df_2014['Total_Pop_Prisional_F'].sum()

Markdown('<strong>{}</strong><br/>{}'.format(title, Total_Pop_Prisional_F))
df_2014['Total_Pop_Prisional_M'] = df_2014['Pop_Prisional_Prov_Just_Est_M'].astype(int)+ df_2014['Pop_Prisional_Prov_Just_Fed_M'].astype(int)+ df_2014['Pop_Prisional_Prov_Outros_M'].astype(int)+ df_2014['Pop_Prisional_Fechado_Just_Est_M'].astype(int)+ df_2014['Pop_Prisional_Fechado_Just_Fed_M'].astype(int)+ df_2014['Pop_Prisional_Fechado_Outros_M'].astype(int)+ df_2014['Pop_Prisional_Semi_Just_Est_M'].astype(int)+ df_2014['Pop_Prisional_Semi_Just_Fed_M'].astype(int)+ df_2014['Pop_Prisional_Semi_Outros_M'].astype(int)+ df_2014['Pop_Prisional_Aberto_Just_Est_M'].astype(int)+ df_2014['Pop_Prisional_Aberto_Just_Fed_M'].astype(int)+ df_2014['Pop_Prisional_Aberto_Outros_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Est_Internacao_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Fed_Internacao_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Outros_Internacao_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Est_Ambulatorial_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Fed_Ambulatorial_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Outros_Ambulatorial_M'].astype(int)
title = "Total da População Prisional Masculina"

Total_Pop_Prisional_M = df_2014['Total_Pop_Prisional_M'].sum()

Markdown('<strong>{}</strong><br/>{}'.format(title, Total_Pop_Prisional_M))
Genero = ['Homens','Mulheres']

Qtde = [Total_Pop_Prisional_M,Total_Pop_Prisional_F]

ax.set_title("População Prisional por Genero", fontsize=18)

ax.set_ylabel("Frequência", fontsize=18);

fig = plt.figure(figsize=(10,6))

plt.bar(Genero,Qtde, color='Blue')

title = "Total da População Prisional em Regime Diferenciado"

Qtde_Regime_Disc_Diferenciado = df_2014['Qtde_Regime_Disc_Diferenciado'].sum().astype(int)

Markdown('<strong>{}</strong><br/>{}'.format(title, Qtde_Regime_Disc_Diferenciado))
df_2014['Total_Gravidas_Lactantes'] = df_2014['Qtde_Gestantes'].astype(int) + df_2014['Qtde_Lactantes'].astype(int)
title = "Total de Grávidas e Lactantes" 

Total_Gravidas_Lactantes = df_2014['Total_Gravidas_Lactantes'].astype(int).sum()

Markdown('<strong>{}</strong><br/>{}'.format(title, Total_Gravidas_Lactantes))
df_2014['Total_Pop_Prisional_Provisoria'] = df_2014['Pop_Prisional_Prov_Just_Est_M'].astype(int)+ df_2014['Pop_Prisional_Prov_Just_Est_F'].astype(int)+ df_2014['Pop_Prisional_Prov_Just_Fed_M'].astype(int)+ df_2014['Pop_Prisional_Prov_Just_Fed_F'].astype(int)+ df_2014['Pop_Prisional_Prov_Outros_M'].astype(int)+ df_2014['Pop_Prisional_Prov_Outros_F'].astype(int)
title = "Total População Provisioria"

Total_Pop_Prisional_Provisoria = df_2014['Total_Pop_Prisional_Provisoria'].sum()

Markdown('<strong>{}</strong><br/>{}'.format(title,Total_Pop_Prisional_Provisoria ))
df_2014['Total_Pop_Prisional_Fechado'] = df_2014['Pop_Prisional_Fechado_Just_Est_M'].astype(int)+ df_2014['Pop_Prisional_Fechado_Just_Est_F'].astype(int)+ df_2014['Pop_Prisional_Fechado_Just_Fed_M'].astype(int)+ df_2014['Pop_Prisional_Fechado_Just_Fed_F'].astype(int)+ df_2014['Pop_Prisional_Fechado_Outros_M'].astype(int)+ df_2014['Pop_Prisional_Fechado_Outros_F'].astype(int)
title = "Total População Regime Fechado"

Total_Pop_Prisional_Fechado = df_2014['Total_Pop_Prisional_Fechado'].sum()

Markdown('<strong>{}</strong><br/>{}'.format(title, Total_Pop_Prisional_Fechado))
df_2014['Total_Pop_Prisional_Semi_Aberto'] = df_2014['Pop_Prisional_Semi_Just_Est_M'].astype(int)+ df_2014['Pop_Prisional_Semi_Just_Est_F'].astype(int)+ df_2014['Pop_Prisional_Semi_Just_Fed_M'].astype(int)+ df_2014['Pop_Prisional_Semi_Just_Fed_F'].astype(int)+ df_2014['Pop_Prisional_Semi_Outros_M'].astype(int)+ df_2014['Pop_Prisional_Semi_Outros_F'].astype(int)
title = "Total Populacão Regime Semi Aberto"

Total_Pop_Prisional_Semi_Aberto = df_2014['Total_Pop_Prisional_Semi_Aberto'].sum()

Markdown('<strong>{}</strong><br/>{}'.format(title, Total_Pop_Prisional_Semi_Aberto))
df_2014['Total_Pop_Prisional_Aberto'] = df_2014['Pop_Prisional_Aberto_Just_Est_M'].astype(int)+ df_2014['Pop_Prisional_Aberto_Just_Est_F'].astype(int)+ df_2014['Pop_Prisional_Aberto_Just_Fed_F'].astype(int)+ df_2014['Pop_Prisional_Aberto_Just_Fed_M'].astype(int)+ df_2014['Pop_Prisional_Aberto_Outros_M'].astype(int)+ df_2014['Pop_Prisional_Aberto_Outros_F'].astype(int)
title = "Total População Regime Aberto"

Total_Pop_Prisional_Aberto = df_2014['Total_Pop_Prisional_Aberto'].sum()

Markdown('<strong>{}</strong><br/>{}'.format(title, Total_Pop_Prisional_Aberto))
df_2014['Total_Pop_Prisional_Med_Seg'] = df_2014['Pop_Prisional_Med_Seg_Just_Est_Internacao_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Est_Internacao_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Fed_Internacao_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Fed_Internacao_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Outros_Internacao_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Outros_Internacao_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Est_Ambulatorial_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Est_Ambulatorial_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Fed_Ambulatorial_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Just_Fed_Ambulatorial_F'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Outros_Ambulatorial_M'].astype(int)+ df_2014['Pop_Prisional_Med_Seg_Outros_Ambulatorial_F'].astype(int)
title = "Total População Medida de Segurança"

Total_Pop_Prisional_Med_Seg = df_2014['Total_Pop_Prisional_Med_Seg'].sum()

Markdown('<strong>{}</strong><br/>{}'.format(title, Total_Pop_Prisional_Med_Seg))
title = "Total População Prisional Regime Diferenciado"

Qtde_Regime_Disc_Diferenciado = df_2014['Qtde_Regime_Disc_Diferenciado'].astype(int).sum()

Markdown('<strong>{}</strong><br/>{}'.format(title,Qtde_Regime_Disc_Diferenciado))
Regime = ['Provisória','Fechado', 'Aberto', 'Semi_Aberto', 'Medidade de Segurança', 'Diferenciado']

Qtde = [Total_Pop_Prisional_Provisoria,Total_Pop_Prisional_Fechado,Total_Pop_Prisional_Aberto,Total_Pop_Prisional_Semi_Aberto,Total_Pop_Prisional_Med_Seg, Qtde_Regime_Disc_Diferenciado]

fig = plt.figure(figsize=(18,6))

plt.bar(Regime,Qtde, color='Green')

plt.title("População Prisional por Regime Prisional", fontsize=18)

plt.xlabel('Categorias')

plt.ylabel('Frequência')
df_2014['Total_Estrangeiros_Retidos'] = df_2014['Qtde_Brasileiros_Naturalizados_M'].astype(int)+df_2014['Qtde_Brasileiros_Naturalizados_F'].astype(int)+df_2014['Qtde_Estrangeiros_M'].astype(int)+df_2014['Qtde_Estrangeiros_F'].astype(int)+df_2014['Qtde_Europa_Alemaes_M'].astype(int)+df_2014['Qtde_Europa_Alemaes_F'].astype(int)+df_2014['Qtde_Europa_Austriacos_M'].astype(int)+df_2014['Qtde_Europa_Austriacos_F'].astype(int)+df_2014['Qtde_Europa_Belgas_M'].astype(int)+df_2014['Qtde_Europa_Europa_Belgas_F'].astype(int)+df_2014['Qtde_Europa_Bulgaros_M'].astype(int)+df_2014['Qtde_Europa_Bulgaros_F'].astype(int)+df_2014['Qtde_Europa_Tchecos_M'].astype(int)+df_2014['Qtde_Europa_Tchecos_F'].astype(int)+df_2014['Qtde_Europa_Croatas_M'].astype(int)+df_2014['Qtde_Europa_Croatas_F'].astype(int)+df_2014['Qtde_Europa_Dinamarqueses_M'].astype(int)+df_2014['Qtde_Europa_Dinamarqueses_F'].astype(int)+df_2014['Qtde_Europa_Escoceses_M'].astype(int)+df_2014['Qtde_Europa_Escoceses_F'].astype(int)+df_2014['Qtde_Europa_Espanhois_M'].astype(int)+df_2014['Qtde_Europa_Europa_Espanhois_F'].astype(int)+df_2014['Qtde_Europa_Franceses_M'].astype(int)+df_2014['Qtde_Europa_Franceses_F'].astype(int)+df_2014['Qtde_Europa_Europa_Gragos_M'].astype(int)+df_2014['Qtde_Europa_Gragos_F'].astype(int)+df_2014['Qtde_Europa_Holandeses_M'].astype(int)+df_2014['Qtde_Europa_Holandeses_F'].astype(int)+df_2014['Qtde_Europa_Hungria_M'].astype(int)+df_2014['Qtde_Europa_Hungria_F'].astype(int)+df_2014['Qtde_Europa_Ingleses_M'].astype(int)+df_2014['Qtde_Europa_Ingleses_F'].astype(int)+df_2014['Qtde_Europa_Irlandeses_M'].astype(int)+df_2014['Qtde_Europa_Irlandeses_F'].astype(int)+df_2014['Qtde_Europa_Italianos_M'].astype(int)+df_2014['Qtde_Europa_Italianos_F'].astype(int)+df_2014['Qtde_Europa_Noruega_M'].astype(int)+df_2014['Qtde_Europa_Noruega_F'].astype(int)+df_2014['Qtde_Europa_Galeses_M'].astype(int)+df_2014['Qtde_Europa_Galeses_F'].astype(int)+df_2014['Qtde_Europa_Poloneses_M'].astype(int)+df_2014['Qtde_Europa_Poloneses_F'].astype(int)+df_2014['Qtde_Europa_Portugueses_M'].astype(int)+df_2014['Qtde_Europa_Portugueses_F'].astype(int)+df_2014['Qtde_Europa_Russos_M'].astype(int)+df_2014['Qtde_Europa_Russos_F'].astype(int)+df_2014['Qtde_Europa_Reino_Unido_M'].astype(int)+df_2014['Qtde_Europa_Reino_Unido_F'].astype(int)+df_2014['Qtde_Europa_Romenos_M'].astype(int)+df_2014['Qtde_Europa_Romenos_F'].astype(int)+df_2014['Qtde_Europa_Servos_M'].astype(int)+df_2014['Qtde_Europa_Servos_F'].astype(int)+df_2014['Qtde_Europa_Sueco_M'].astype(int)+df_2014['Qtde_Europa_Sueco_F'].astype(int)+df_2014['Qtde_Europa_Suicos_M'].astype(int)+df_2014['Qtde_Europa_Suicos_F'].astype(int)+df_2014['Qtde_Europa_Outros_Paises_M'].astype(int)+df_2014['Qtde_Europa_Outros_Paises_F'].astype(int)+df_2014['Qtde_Asia_Afeganistao_M'].astype(int)+df_2014['Qtde_Asia_Afeganistao_F'].astype(int)+df_2014['Qtde_Asia_Arabia_Saudita_M'].astype(int)+df_2014['Qtde_Asia_Arabia_Saudita_F'].astype(int)+df_2014['Qtde_Asia_Catar_M'].astype(int)+df_2014['Qtde_Asia_Catar_F'].astype(int)+df_2014['Qtde_Asia_Cazaquistao_M'].astype(int)+df_2014['Qtde_Asia_Cazaquistao_F'].astype(int)+df_2014['Qtde_Asia_China_M'].astype(int)+df_2014['Qtde_Asia_China_F'].astype(int)+df_2014['Qtde_Asia_Coreia_Norte_M'].astype(int)+df_2014['Qtde_Asia_Coreia_Norte_F'].astype(int)+df_2014['Qtde_Asia_Coreia_Sul_M'].astype(int)+df_2014['Qtde_Asia_Coreia_Sul_F'].astype(int)+df_2014['Qtde_Asia_Emirados_Arabes_M'].astype(int)+df_2014['Qtde_Asia_Emirados_Arabes_F'].astype(int)+df_2014['Qtde_Asia_Filipinas_M'].astype(int)+df_2014['Qtde_Asia_Filipinas_F'].astype(int)+df_2014['Qtde_Asia_India_M'].astype(int)+df_2014['Qtde_Asia_India_F'].astype(int)+df_2014['Qtde_Asia_Indonesia_M'].astype(int)+df_2014['Qtde_Asia_Indonesia_F'].astype(int)+df_2014['Qtde_Asia_Ira_M'].astype(int)+df_2014['Qtde_Asia_Ira_F'].astype(int)+df_2014['Qtde_Asia_Iraque_M'].astype(int)+df_2014['Qtde_Asia_Iraque_F'].astype(int)+df_2014['Qtde_Asia_Irael_M'].astype(int)+df_2014['Qtde_Asia_Irael_F'].astype(int)+df_2014['Qtde_Asia_Japao_M'].astype(int)+df_2014['Qtde_Asia_Japao_F'].astype(int)+df_2014['Qtde_Asia_Jordania_M'].astype(int)+df_2014['Qtde_Asia_Jordania_F'].astype(int)+df_2014['Qtde_Asia_Kuwait_M'].astype(int)+df_2014['Qtde_Asia_Kuwait_F'].astype(int)+df_2014['Qtde_Asia_Libano_M'].astype(int)+df_2014['Qtde_Asia_Libano_F'].astype(int)+df_2014['Qtde_Asia_Libano_M'].astype(int)+df_2014['Qtde_Asia_Libano_F'].astype(int)+df_2014['Qtde_Asia_Malasia_M'].astype(int)+df_2014['Qtde_Asia_Malasia_F'].astype(int)+df_2014['Qtde_Asia_Paquistao_M'].astype(int)+df_2014['Qtde_Asia_Paquistao_F'].astype(int)+df_2014['Qtde_Asia_Siria_M'].astype(int)+df_2014['Qtde_Asia_Siria_F'].astype(int)+df_2014['Qtde_Asia_Sri_Lanka_M'].astype(int)+df_2014['Qtde_Asia_Sri_Lanka_F'].astype(int)+df_2014['Qtde_Asia_Tailandia_M'].astype(int)+df_2014['Qtde_Asia_Tailandia_F'].astype(int)+df_2014['Qtde_Asia_Taiwan_M'].astype(int)+df_2014['Qtde_Asia_Taiwan_F'].astype(int)+df_2014['Qtde_Asia_Turquia_M'].astype(int)+df_2014['Qtde_Asia_Turquia_F'].astype(int)+df_2014['Qtde_Asia_Timor_Leste_M'].astype(int)+df_2014['Qtde_Asia_Timor_Leste_F'].astype(int)+df_2014['Qtde_Asia_Vietna_M'].astype(int)+df_2014['Qtde_Asia_Vietna_F'].astype(int)+df_2014['Qtde_Asia_Outros_Paises_M'].astype(int)+df_2014['Qtde_Asia_Outros_Paises_F'].astype(int)+df_2014['Qtde_Africa_Africa_Do_Sul_M'].astype(int)+df_2014['Qtde_Africa_Africa_Do_Sul_F'].astype(int)+df_2014['Qtde_Africa_Angola_M'].astype(int)+df_2014['Qtde_Africa_Angola_F'].astype(int)+df_2014['Qtde_Africa_Argelia_M'].astype(int)+df_2014['Qtde_Africa_Argelia_F'].astype(int)+df_2014['Qtde_Africa_Cabo_Verde_M'].astype(int)+df_2014['Qtde_Africa_Cabo_Verde_F'].astype(int)+df_2014['Qtde_Africa_Camaroes_M'].astype(int)+df_2014['Qtde_Africa_Camaroes_F'].astype(int)+df_2014['Qtde_Africa_Republica_Do_Congo_M'].astype(int)+df_2014['Qtde_Africa_Republica_Do_Congo_F'].astype(int)+df_2014['Qtde_Africa_Costa_Do_Marfim_M'].astype(int)+df_2014['Qtde_Africa_Costa_Do_Marfim_F'].astype(int)+df_2014['Qtde_Africa_Egito_M'].astype(int)+df_2014['Qtde_Africa_Egito_F'].astype(int)+df_2014['Qtde_Africa_Etiopia_M'].astype(int)+df_2014['Qtde_Africa_Etiopia_F'].astype(int)+df_2014['Qtde_Africa_Gana_M'].astype(int)+df_2014['Qtde_Africa_Gana_F'].astype(int)+df_2014['Qtde_Africa_Guine_M'].astype(int)+df_2014['Qtde_Africa_Guine_F'].astype(int)+df_2014['Qtde_Africa_Guine_Bissau_M'].astype(int)+df_2014['Qtde_Africa_Guine_Bissau_F'].astype(int)+df_2014['Qtde_Africa_Libia_M'].astype(int)+df_2014['Qtde_Africa_Libia_F'].astype(int)+df_2014['Qtde_Africa_Madagascar_M'].astype(int)+df_2014['Qtde_Africa_Madagascar_F'].astype(int)+df_2014['Qtde_Africa_Marrocos_M'].astype(int)+df_2014['Qtde_Africa_Marrocos_F'].astype(int)+df_2014['Qtde_Africa_Mocambique_M'].astype(int)+df_2014['Qtde_Africa_Mocambique_F'].astype(int)+df_2014['Qtde_Africa_Nigeria_M'].astype(int)+df_2014['Qtde_Africa_Nigeria_F'].astype(int)+df_2014['Qtde_Africa_Quenia_M'].astype(int)+df_2014['Qtde_Africa_Quenia_F'].astype(int)+df_2014['Qtde_Africa_Ruanda_M'].astype(int)+df_2014['Qtde_Africa_Ruanda_F'].astype(int)+df_2014['Qtde_Africa_Senegal_M'].astype(int)+df_2014['Qtde_Africa_Senegal_F'].astype(int)+df_2014['Qtde_Africa_Serra_Leoa_M'].astype(int)+df_2014['Qtde_Africa_Serra_Leoa_F'].astype(int)+df_2014['Qtde_Africa_Somalia_M'].astype(int)+df_2014['Qtde_Africa_Somalia_F'].astype(int)+df_2014['Qtde_Africa_Tunisia_M'].astype(int)+df_2014['Qtde_Africa_Tunisia_F'].astype(int)+df_2014['Qtde_Africa_Outros_Paises_M'].astype(int)+df_2014['Qtde_Africa_Outros_Paises_F'].astype(int)+df_2014['Qtde_America_Argentina_M'].astype(int)+df_2014['Qtde_America_Argentina_F'].astype(int)+df_2014['Qtde_America_Bolivia_M'].astype(int)+df_2014['Qtde_America_Bolivia_F'].astype(int)+df_2014['Qtde_America_Canada_M'].astype(int)+df_2014['Qtde_America_Canada_F'].astype(int)+df_2014['Qtde_America_Chile_M'].astype(int)+df_2014['Qtde_America_Chile_F'].astype(int)+df_2014['Qtde_America_Colombia_M'].astype(int)+df_2014['Qtde_America_Colombia_F'].astype(int)+df_2014['Qtde_America_Costa_Rica_M'].astype(int)+df_2014['Qtde_America_Costa_Rica_F'].astype(int)+df_2014['Qtde_America_Cuba_M'].astype(int)+df_2014['Qtde_America_Cuba_F'].astype(int)+df_2014['Qtde_America_El_Salvador_M'].astype(int)+df_2014['Qtde_America_El_Salvador_F'].astype(int)+df_2014['Qtde_America_Equador_M'].astype(int)+df_2014['Qtde_America_Equador_F'].astype(int)+df_2014['Qtde_America_EUA_M'].astype(int)+df_2014['Qtde_America_EUA_F'].astype(int)+df_2014['Qtde_America_Quatemala_M'].astype(int)+df_2014['Qtde_America_Quatemala_F'].astype(int)+df_2014['Qtde_America_Quiana_M'].astype(int)+df_2014['Qtde_America_Quiana_F'].astype(int)+df_2014['Qtde_America_Quiana_Francesa_M'].astype(int)+df_2014['Qtde_America_Quiana_Francesa_F'].astype(int)+df_2014['Qtde_America_Haiti_M'].astype(int)+df_2014['Qtde_America_Haiti_F'].astype(int)+df_2014['Qtde_America_Honduras_M'].astype(int)+df_2014['Qtde_America_Honduras_F'].astype(int)+df_2014['Qtde_America_Ilhas_Cayman_M'].astype(int)+df_2014['Qtde_America_Ilhas_Cayman_F'].astype(int)+df_2014['Qtde_America_Jamaica_M'].astype(int)+df_2014['Qtde_America_Jamaica_F'].astype(int)+df_2014['Qtde_America_Mexico_M'].astype(int)+df_2014['Qtde_America_Mexico_F'].astype(int)+df_2014['Qtde_America_Nicaragua_M'].astype(int)+df_2014['Qtde_America_Nicaragua_F'].astype(int)+df_2014['Qtde_America_Panama_M'].astype(int)+df_2014['Qtde_America_Panama_F'].astype(int)+df_2014['Qtde_America_Paraguai_M'].astype(int)+df_2014['Qtde_America_Paraguai_F'].astype(int)+df_2014['Qtde_America_Peru_M'].astype(int)+df_2014['Qtde_America_Peru_F'].astype(int)+df_2014['Qtde_America_Porto_Rico_M'].astype(int)+df_2014['Qtde_America_Porto_Rico_F'].astype(int)+df_2014['Qtde_America_Republica_Dominicana_M'].astype(int)+df_2014['Qtde_America_Republica_Dominicana_F'].astype(int)+df_2014['Qtde_America_Suriname_M'].astype(int)+df_2014['Qtde_America_Suriname_F'].astype(int)+df_2014['Qtde_America_Trinidad_Tobago_M'].astype(int)+df_2014['Qtde_America_Trinidad_Tobago_F'].astype(int)+df_2014['Qtde_America_Uruguai_M'].astype(int)+df_2014['Qtde_America_Uruguai_F'].astype(int)+df_2014['Qtde_America_Venezuela_M'].astype(int)+df_2014['Qtde_America_Venezuela_F'].astype(int)+df_2014['Qtde_America_Outros_Paises_M'].astype(int)+df_2014['Qtde_America_Outros_Paises_F'].astype(int)+df_2014['Qtde_Oceania_Australia_M'].astype(int)+df_2014['Qtde_Oceania_Australia_F'].astype(int)+df_2014['Qtde_Oceania_Nova_Zelandia_M'].astype(int)+df_2014['Qtde_Oceania_Nova_Zelandia_F'].astype(int)+df_2014['Qtde_Oceania_Outros_Paises_M'].astype(int)+df_2014['Qtde_Oceania_Outros_Paises_F'].astype(int)+df_2014['Qtde_Sem_Nacionalidade_M'].astype(int)+df_2014['Qtde_Sem_Nacionalidade_F'].astype(int)
df_2014['Total_Estrangeiros_Retidos'].fillna(0)

title = "Total de Estrangeiros Retidos"

Total_Estrangeiros_Retidos = df_2014['Total_Estrangeiros_Retidos'].sum()

Markdown('<strong>{}</strong><br/>{}'.format(title, Total_Estrangeiros_Retidos))
df_2014['Total_Pop_Prisional'].plot.kde()
df_2014['Capacidade_Pop_Prisional'].plot.kde()
perc_prov = (df_2014['Total_Pop_Prisional_Fechado'].sum() / df_2014['Total_Pop_Prisional'].sum())*100

title = "Percentual de População Prisional Regime Fechado"

Markdown('<strong>{}</strong><br/>{}'.format(title, perc_prov))
perc_prov = (df_2014['Total_Pop_Prisional_Provisoria'].sum() / df_2014['Total_Pop_Prisional'].sum())*100

title = "Percentual de População Prisional Provisoria"

Markdown('<strong>{}</strong><br/>{}'.format(title, perc_prov))
perc_prov = (df_2014['Total_Pop_Prisional_F'].sum() / df_2014['Total_Pop_Prisional'].sum())*100

title = "Percentual de População Prisional Feminina"

Markdown('<strong>{}</strong><br/>{}'.format(title, perc_prov))
perc_prov = (df_2014['Total_Pop_Prisional_M'].sum() / df_2014['Total_Pop_Prisional'].sum())*100

title = "Percentual de População Prisional Masculina"

Markdown('<strong>{}</strong><br/>{}'.format(title, perc_prov))
perc_prov = (df_2014['Total_Estrangeiros_Retidos'].sum() / df_2014['Total_Pop_Prisional'].sum())*100

title = "Percentual de População Prisional de Estrangeiros"

Markdown('<strong>{}</strong><br/>{}'.format(title, perc_prov))
df_2014_UF['Unidade_Prisional'].groupby(df_2014_UF['UF']).agg('count').sort_values(ascending=False)
df_UF = df_2014_UF['Unidade_Prisional'].groupby(df_2014_UF['UF']).agg('count').sort_values(ascending=False)

fig = plt.figure(figsize=(18,6))

df_UF.plot(kind = 'bar' )
ax = df_2014_UF['UF'].value_counts().plot(kind='bar', figsize=(22,7),

                                        color="coral", fontsize=13);

ax.set_alpha(0.8)

ax.set_title("Unidades Federativas", fontsize=18)

ax.set_ylabel("Percentual de Unidades Prisionais", fontsize=18);



totals = []

for i in ax.patches:

    totals.append(i.get_height())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_x()-.03, i.get_height()+.5, \

            str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,

                color='dimgrey')
df_2014['Total_Pop_Prisional_Raca'] = df_2014['Qtde_Brancos_M'].astype(int)+df_2014['Qtde__Brancos_F'].astype(int)+df_2014['Qtde_Negros_M'].astype(int)+df_2014['Qtde_Negros_F'].astype(int)+df_2014['Qtde_Pardos_M'].astype(int)+df_2014['Qtde_Pardos_F'].astype(int)+df_2014['Qtde_Amarelos_M'].astype(int)+df_2014['Qtde_Amarelos_F'].astype(int)+df_2014['Qtde_Indigenas_M'].astype(int)+df_2014['Qtde_Indigenas_F'].astype(int)+df_2014['Qtde_Outros_Racas_M'].astype(int)+df_2014['Qtde_Outros_Racas_F'].astype(int)+df_2014['Qtde_Nao_Inf_Racas_M'].astype(int)+df_2014['Qtde_Nao_Inf_Racas_F'].astype(int)
Qtde_Brancos = df_2014['Qtde_Brancos_M'].sum().astype(int)+df_2014['Qtde__Brancos_F'].sum().astype(int)

Qtde_Negros = df_2014['Qtde_Negros_M'].sum().astype(int)+df_2014['Qtde_Negros_F'].sum().astype(int)

Qtde_Pardos = df_2014['Qtde_Pardos_M'].sum().astype(int)+df_2014['Qtde_Pardos_F'].sum().astype(int)

Qtde_Amarelos = df_2014['Qtde_Amarelos_M'].sum().astype(int)+df_2014['Qtde_Amarelos_F'].sum().astype(int)

Qtde_Indigenas = df_2014['Qtde_Indigenas_M'].sum().astype(int)+df_2014['Qtde_Indigenas_F'].sum().astype(int)

Qtde_Outros = df_2014['Qtde_Outros_Racas_M'].sum().astype(int)+df_2014['Qtde_Outros_Racas_F'].sum().astype(int)

Qtde_Nao_Informado = df_2014['Qtde_Nao_Inf_Racas_M'].sum().astype(int)+df_2014['Qtde_Nao_Inf_Racas_F'].sum().astype(int)
Raca = ['Brancos','Negros','Pardos','Amarelos','Indigenas','Outros','Não Informado']

Qtde = [Qtde_Brancos,Qtde_Negros,Qtde_Pardos,Qtde_Amarelos,Qtde_Indigenas,Qtde_Outros,Qtde_Nao_Informado]

fig = plt.figure(figsize=(18,6))

plt.bar(Raca,Qtde, color='orange')

ax.set_title("População Prisional por Raça", fontsize=18)

ax.set_ylabel("Frequência", fontsize=18);
df_ = df_2014.groupby(['UF']).Total_Pop_Prisional.agg('sum').to_frame('Qtde_Pop_Por_Estado').reset_index() 
df_
df_.sum()
fig = plt.figure(figsize=(45,10))

plt.bar(df_['UF'],df_['Qtde_Pop_Por_Estado'], color='red')

ax.set_title("População Prisional por Estado", fontsize=18,)

ax.set_ylabel("Frequência", fontsize=18);
m = folium.Map(

    location=[-14.235, -51.9253],

    popup='Brasil',

    zoom_start=4.5,

    tiles= "OpenStreetMap"

    #tiles = "Stamen Terrain",

)

#Acre

folium.Circle(

    location=[-8.77, -70.55],

    popup='Acre',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 3488

).add_to(m)

#Alagoas

folium.Circle(

    location=[-9.71, -35.73],

    popup='Alagoas',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 5423

).add_to(m)



#Amazonas

folium.Circle(

    location=[-3.07, -61.66],

    popup='Amazonas',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 2654

).add_to(m)



#Amapa

folium.Circle(

    location=[1.41, -51.77],

    popup='Amapá',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 7378

).add_to(m)



#Bahia

folium.Circle(

    location=[-12.96, -38.51],

    popup='Bahia',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 11848

).add_to(m)



#Ceara

folium.Circle(

    location=[-3.71, -38.54],

    popup='Ceará',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 20101

).add_to(m)



#DF

folium.Circle(

    location=[-15.83, -47.86],

    popup='Distrito Federal',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 13269

).add_to(m)



#Espirito Santos

folium.Circle(

    location=[-19.19, -40.34],

    popup='Espirito Santo',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 15461

).add_to(m)



#Goias

folium.Circle(

    location=[-16.64, -49.31],

    popup='Goiás',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 13286

).add_to(m)



#Maranhao

folium.Circle(

    location=[-2.55, -44.30],

    popup='Maranhão',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 4938

).add_to(m)



#Mato Grosso

folium.Circle(

    location=[-12.64, -55.42],

    popup='Mato Grosso',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 10533

).add_to(m)



#Mato Grosso do Sul

folium.Circle(

    location=[-20.51, -54.54],

    popup='Mato Grosso do Sul',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 13125

).add_to(m)



#Minas Gerais

folium.Circle(

    location=[-18.10, -44.38],

    popup='Minas Gerais',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 57005

).add_to(m)



#Pará

folium.Circle(

    location=[-5.53, -52.29],

    popup='Pará',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 19518

).add_to(m)



#Paraíba

folium.Circle(

    location=[-7.06, -35.55],

    popup='Paraíba',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 9606

).add_to(m)



#Paraná

folium.Circle(

    location=[-24.89, -51.55],

    popup='Paraná',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 13493

).add_to(m)



#Pernanmbuco

folium.Circle(

    location=[-8.28, -35.07],

    popup='Pernanmbuco',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 31510

).add_to(m)



#Piauí

folium.Circle(

    location=[-8.28, -43.68],

    popup='Piauí',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 2414

).add_to(m)



#Rio de Janeiro

folium.Circle(

    location=[-22.84, -43.15],

    popup='Rio de Janeiro',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 7203

).add_to(m)



#Rio Grande do Norte

folium.Circle(

    location=[-5.22, -36.52],

    popup='Rio Grande do Norte',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 28233

).add_to(m)



#Rondonia

folium.Circle(

    location=[-11.22, -62.80],

    popup='Rondonia',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 45696

).add_to(m)



#Rio Grande do Sul

folium.Circle(

    location=[-30.01, -51.22],

    popup='Rio Grande do Sul',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 7616

).add_to(m)



#Roraima

folium.Circle(

    location=[1.89, -61.22],

    popup='Roraima',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 1609

).add_to(m)



#Santa Catarina

folium.Circle(

    location=[-27.33, -49.44],

    popup='',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 17005

).add_to(m)



#Sergipe

folium.Circle(

    location=[-10.90, -37.07],

    popup='Sergipe',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 4057

).add_to(m)



#São Paulo

folium.Circle(

    location=[-23.55, -46.64],

    popup='São Paulo',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 214843

).add_to(m)



#Tocantins

folium.Circle(

    location=[-10.25, -48.25],

    popup='Tocantins',

    icon=folium.Icon(icon='map-marker-alt'),

    color='crimson',

    fill=True,

    fill_color='crimson',

    radius = 3254

).add_to(m)

m

df_2014['Deficit_Pop_Prisional'] = (df_2014['Total_Pop_Prisional'].astype(int) / df_2014['Capacidade_Pop_Prisional'].astype(int))*100
df_2014['Deficit_Pop_Prisional']
from numpy import inf

df_2014[df_2014['Deficit_Pop_Prisional'] == inf] = 0

df_2014.update(df_2014.fillna(0))
df_deficit = df_2014.groupby(['UF']).Total_Pop_Prisional.agg('sum').to_frame('Total_Pop_Prisional').reset_index() 
df_deficit1 = df_2014.groupby(['UF']).Capacidade_Pop_Prisional.agg('sum').to_frame('Capacidade_Pop_Prisional').reset_index() 
df_deficit['Capacidade_Pop_Prisional'] = df_deficit1['Capacidade_Pop_Prisional']
df_deficit['Deficit_Vagas'] = df_deficit['Total_Pop_Prisional'].astype(int) - df_deficit['Capacidade_Pop_Prisional'].astype(int)
df_deficit['Super_Lotacao'] = df_deficit['Total_Pop_Prisional'].astype(int) / df_deficit['Capacidade_Pop_Prisional'].astype(int)*100
df_deficit
df_2014.drop(['Pop_Prisional_Prov_Just_Est_M' ,'Pop_Prisional_Prov_Just_Est_F' ,'Pop_Prisional_Prov_Just_Fed_M' ,'Pop_Prisional_Prov_Just_Fed_F' ,'Pop_Prisional_Prov_Outros_M' ,'Pop_Prisional_Prov_Outros_F' ,'Pop_Prisional_Fechado_Just_Est_M' ,'Pop_Prisional_Fechado_Just_Est_F' ,'Pop_Prisional_Fechado_Just_Fed_M' ,'Pop_Prisional_Fechado_Just_Fed_F' ,'Pop_Prisional_Fechado_Outros_M' ,'Pop_Prisional_Fechado_Outros_F' ,'Pop_Prisional_Semi_Just_Est_M' ,'Pop_Prisional_Semi_Just_Est_F' ,'Pop_Prisional_Semi_Just_Fed_M' ,'Pop_Prisional_Semi_Just_Fed_F' ,'Pop_Prisional_Semi_Outros_M' ,'Pop_Prisional_Semi_Outros_F' ,'Pop_Prisional_Aberto_Just_Est_M' ,'Pop_Prisional_Aberto_Just_Est_F' ,'Pop_Prisional_Aberto_Just_Fed_F' ,'Pop_Prisional_Aberto_Just_Fed_M' ,'Pop_Prisional_Aberto_Outros_M' ,'Pop_Prisional_Aberto_Outros_F' ,'Pop_Prisional_Med_Seg_Just_Est_Internacao_M' ,'Pop_Prisional_Med_Seg_Just_Est_Internacao_F' ,'Pop_Prisional_Med_Seg_Just_Fed_Internacao_M' ,'Pop_Prisional_Med_Seg_Just_Fed_Internacao_F' ,'Pop_Prisional_Med_Seg_Outros_Internacao_M' ,'Pop_Prisional_Med_Seg_Outros_Internacao_F' ,'Pop_Prisional_Med_Seg_Just_Est_Ambulatorial_M' ,'Pop_Prisional_Med_Seg_Just_Est_Ambulatorial_F' ,'Pop_Prisional_Med_Seg_Just_Fed_Ambulatorial_M' ,'Pop_Prisional_Med_Seg_Just_Fed_Ambulatorial_F' ,'Pop_Prisional_Med_Seg_Outros_Ambulatorial_M' ,'Pop_Prisional_Med_Seg_Outros_Ambulatorial_F','Qtde_Regime_Disc_Diferenciado'], axis=1)
df_2014.drop(['Capacidade_Prov_M','Capacidade_Prov_F','Capacidade_Fechado_M','Capacidade_Fechado_F','Capacidade_Semi_M','Capacidade_Semi_F','Capacidade_Aberto_M','Capacidade_Aberto_F','Capacidade_RDD_M','Capacidade_RDD_F','Capacidade_Intern_M','Capacidade_Intern_F','Capacidade_Outros_M','Capacidade_Outros_F','Capacidade_Outros'], axis=1)
df_2014_new = df_2014[['Capacidade_Prov_M','Capacidade_Prov_F','Capacidade_Fechado_M','Capacidade_Fechado_F','Capacidade_Semi_F','Capacidade_Semi_M','Capacidade_Aberto_M','Capacidade_Aberto_F','Capacidade_RDD_M','Capacidade_RDD_F','Capacidade_Intern_M','Capacidade_Intern_F','Capacidade_Outros_M','Capacidade_Outros_F','Celas_N_Aptas','Celas_N_Aptas_M','Celas_N_Aptas_F','Alimentacao_Terc','Limpeza_Terc','Lavanderia_Terc','Saude_Terc','Seguranca_Terc','Assist_Educ_Terc','Assist_Laboral_Terc','Assist_Social_Terc','Assist_Jur_Terc','Serv_Adm_Terc','Serv_Terc_Outros','Qtde_Farmacia','Qtde_Salas_Aula','Capacidade_Sala_Aula','Capacidade_Sala_Encontro','Qtde_Sala_Professores','Sala_Professores_Capacidade','Oficina_Carga_Descarga','Oficina_Concreto_Capacidade','Oficina_Blocos_Capacidade','Oficina_Padaria_Capacidade','Oficina_Artesanato_Capacidade','Oficina_Marcenaria_Capacidade','Oficina_Serralheria_Capacidade','Oficina_Capacidade_Outros1','Oficina_Capacidade_Outros2','Oficina_Capacidade_Outros3','Oficina_Capacidade_Outros4','Oficina_Capacidade_Outros5','Oficina_Capacidade_Outros6','Oficina_Capacidade_Outros8','Oficina_Capacidade_Outros9','Oficina_Capacidade_Outros10','Qtde_Cargo_Adm_M','Qtde_Cargo_Adm_F','Qtde_Cargo_Comissionado_M','Qtde_Cargo_Comissionado_F','Qtde_Cargo_Adm_Terc_M','Qtde_Cargo_Adm_Terc_F','Qtde_Cargo_Adm_Temp_M','Qtde_Cargo_Adm_Temp_F','Qtde_Enfermeiros','Qtde_Enfermeiras','Enfermeiros_Comissionado','Qtde_Enfermeiras_Comissionadas','Qtde_Enfermeiros_Terc','Qtde_Enfermeiras_Terc','Qtde_Enfermeiros_Temp','Qtde_Enfermeiras_Temp','Qtde_Tec_Enfermagem_M','Qtde_Tec_Enfermagem_F','Qtde_Tec_Enfermagem_Comissionado_M','Qtde_Tec_Enfermagem_Comissionado_F','Qtde_Tec_Enfermagem_Terc_M','Qtde_Tec_Enfermagem_Terc_F','Qtde_Tec_Enfermagem_Temp_M','Qtde_Tec_Enfermagem_Temp_F','Qtde_Psicologos','Qtde_Psicologas','Qtde_Psicologos_Comissionados','Qtde_Psicologas_Comissionadas','Qtde_Psicologos_Terc','Qtde_Psicologas_Terc','Qtde_Psicologos_Temp','Qtde_Psicologas_Temp','Qtde_Dentista_M','Qtde_Dentista_F','Qtde_Dentista_Comissionado_M','Qtde_Dentista_Comissionado_F','Qtde_Denstista_Terc_M','Qtde_Denstista_Terc_F','Qtde_Dentista_Temp_M','Qtde_Dentista_Temp_F','Qtde_Aux_Odont_M','Qtde_Aux_Odont_F','Qtde_Aux_Odont_Comissionado_M','Qtde_Aux_Odont_Comissionado_F','Qtde_Aux_Odont_Terc_M','Qtde_Aux_Odont_Terc_F','Qtde_Aux_Odont_Temp_M','Qtde_Aux_Odont_Temp_F','Qtde_Assist_Social_M','Qtde_Assist_Social_F','Qtde_Assist_Social_Comissionado_M','Qtde_Assist_Social_Comissionado_F','Qtde_Assist_Social_Terc_M','Qtde_Assist_Social_Terc_F','Qtde_Assist_Social_Temp_M','Qtde_Assist_Social_Temp_F','Qtde_Advogados','Qtde_Advogadas','Qtde_Advogados_Comissionados','Qtde_Advogadas_Comissionadas','Qtde_Advogados_Terc','Qtde_Advogadas_Terc','Qtde_Advogados_Temp','Qtde_Advogadas_Temp','Qtde_Medicos','Qtde_Medicas','Qtde_Medicos_Comissionados','Qtde_Medicas_Comissionados','Qtde_Medicos_Terc','Qtde_Medicas_Terc','Qtde_Medicos_Temp','Qtde_Medicas_Temp','Qtde_Ginecologistas_M','Qtde_Ginecologistas_F','Qtde_Ginecologistas_Comissionados_M','Qtde_Ginecologistas_Comissionados_F','Qtde_Ginecologistas_Terc_M','Qtde_Ginecologistas_Terc_F','Qtde_Ginecologitas_Temp_M','Qtde_Ginecologitas_Temp_F','Qtde_Psiquiatras_M','Qtde_Psiquiatras_F','Qtde_Psiquiatras_Comissionados_M','Qtde_Psiquiatras_Comissionados_F','Qtde_Psiquiatras_Terc_M','Qtde_Psiquiatras_Terc_F','Qtde_Psiquiatras_Temp_M','Qtde_Psiquiatras_Temp_F','Qtde_Medicos_Outros_M','Qtde_Medicos_Outros_F','Qtde_Medicos_Outros_Comissionados_M','Qtde_Medicos_Outros_Comissionados_F','Qtde_Medicos_Outros_Terc_M','Qtde_Medicos_Outros_Terc_F','Qtde_Medicos_Outros_Temp_M','Qtde_Medicos_Outros_Temp_F','Qtde_Pedagogos_M','Qtde_Pedagogos_F','Qtde_Pedagogos_Comissionados_M','Qtde_Pedagogos_Comissionados_F','Qtde_Pedagogos_Terc_M','Qtde_Pedagogos_Terc_F','Qtde_Pedagogos_Temp_M','Qtde_Pedagogos_Temp_F','Qtde_Professores_M','Qtde_Professores_F','Qtde_Professores_Comissionados_M','Qtde_Professores_Comissionados_F','Qtde_Professores_Terc_M','Qtde_Professores_Terc_F','Qtde_Professores_Temp_M','Qtde_Professores_Temp_F','Qtde_Terapeutas_M','Qtde_Terapeutas_F','Qtde_Terapeutas_Comissionados_M','Qtde_Terapeutas_Comissionados_F','Qtde_Terapeutas_Terc_M','Qtde_Terapeutas_Terc_F','Qtde_Terapeutas_Temp_M','Qtde_Terapeutas_Temp_F','Qtde_PM_M','Qtde_PM_F','Qtde_PC_Comissionado_M','Qtde_PC_Comissionado_F','Qtde_PC_Terc_M','Qtde_PC_Terc_F','Qtde_PC_Temp_M','Qtde_PC_Temp_F','Qtde_PM_M','Qtde_PM_F','Qtde_PM_Comissionado_M','Qtde_PM_Comissionado_F','Qtde_PM_Terc_M','Qtde_PM_Terc_F','Qtde_PM_Temp_M','Qtde_PM_Temp_F','Qtde_Servidores_Outros_M','Qtde_Servidores_Outros_F','Qtde_Servidores_Outros_Comissionados_M','Qtde_Servidores_Outros_Comissionados_F','Qtde_Servidores_Outros_Terc_M','Qtde_Servidores_Outros_Terc_F','Qtde_Servidores_Outros_Temp_M','Qtde_Servidores_Outros_Temp_F','Cargos_Outros1_M','Cargos_Outros1_F','Cargos_Outros1_Comissionado_M','Cargos_Outros1_Comissionado_F','Cargos_Outros1_Terc_M','Cargos_Outros1_Terc_F','Cargos_Outros1_Temp_M','Cargos_Outros1_Temp_F','Cargos_Outros2_M','Cargos_Outros2_F','Cargos_Outros2_Comissionado_M','Cargos_Outros2_Comissionado_F','Cargos_Outros2_Terc_M','Cargos_Outros2_Terc_F','Cargos_Outros2_Temp_M','Cargos_Outros2_Temp_F','Cargos_Outros3_M','Cargos_Outros1_F','Cargos_Outros3_Comissionado_M','Cargos_Outros3_Comissionado_F','Cargos_Outros3_Terc_M','Cargos_Outros3_Terc_F','Cargos_Outros3_Temp_M','Cargos_Outros3_Temp_F','Cargos_Outros4_M','Cargos_Outros4_F','Cargos_Outros4_Comissionado_M','Cargos_Outros4_Comissionado_F','Cargos_Outros4_Terc_M','Cargos_Outros4_Terc_F','Cargos_Outros4_Temp_M','Cargos_Outros4_Temp_F','Cargos_Outros5_M','Cargos_Outros5_F','Cargos_Outros5_Comissionado_M','Cargos_Outros5_Comissionado_F','Cargos_Outros5_Terc_M','Cargos_Outros5_Terc_F','Cargos_Outros5_Temp_M','Cargos_Outros5_Temp_F','Cargos_Outros6_M','Cargos_Outros6_F','Cargos_Outros6_Comissionado_M','Cargos_Outros6_Comissionado_F','Cargos_Outros6_Terc_M','Cargos_Outros6_Terc_F','Cargos_Outros6_Temp_M','Cargos_Outros6_Temp_F','Cargos_Outros7_M','Cargos_Outros7_F','Cargos_Outros7_Comissionado_M','Cargos_Outros7_Comissionado_F','Cargos_Outros7_Terc_M','Cargos_Outros7_Terc_F','Cargos_Outros7_Temp_M','Cargos_Outros7_Temp_F','Cargos_Outros8_M','Cargos_Outros8_F','Cargos_Outros8_Comissionado_M','Cargos_Outros8_Comissionado_F','Cargos_Outros8_Terc_M','Cargos_Outros8_Terc_F','Cargos_Outros8_Temp_M','Cargos_Outros8_Temp_F','Cargos_Outros9_M','Cargos_Outros9_F','Cargos_Outros9_Comissionado_M','Cargos_Outros9_Comissionado_F','Cargos_Outros9_Terc_M','Cargos_Outros9_Terc_F','Cargos_Outros9_Temp_M','Cargos_Outros9_Temp_F','Cargos_Outros10','Cargos_Outros10_M','Cargos_Outros10_F','Cargos_Outros10_Comissionado_M','Cargos_Outros10_Comissionado_F','Cargos_Outros10_Terc_M','Cargos_Outros10_Terc_F','Cargos_Outros10_Temp_M','Cargos_Outros10_Temp_F','Assist_Juridica_Outro','Pop_Prisional_Prov_Just_Est_M','Pop_Prisional_Prov_Just_Est_F','Pop_Prisional_Prov_Just_Fed_M','Pop_Prisional_Prov_Just_Fed_F','Pop_Prisional_Prov_Outros_M','Pop_Prisional_Prov_Outros_F','Pop_Prisional_Fechado_Just_Est_M','Pop_Prisional_Fechado_Just_Est_F','Pop_Prisional_Fechado_Just_Fed_M','Pop_Prisional_Fechado_Just_Fed_F','Pop_Prisional_Fechado_Outros_M','Pop_Prisional_Fechado_Outros_F','Pop_Prisional_Semi_Just_Est_M','Pop_Prisional_Semi_Just_Est_F','Pop_Prisional_Semi_Just_Fed_M','Pop_Prisional_Semi_Just_Fed_F','Pop_Prisional_Semi_Outros_M','Pop_Prisional_Semi_Outros_F','Pop_Prisional_Aberto_Just_Est_M','Pop_Prisional_Aberto_Just_Est_F','Pop_Prisional_Aberto_Just_Fed_F','Pop_Prisional_Aberto_Just_Fed_M','Pop_Prisional_Aberto_Outros_M','Pop_Prisional_Aberto_Outros_F','Pop_Prisional_Med_Seg_Just_Est_Internacao_M','Pop_Prisional_Med_Seg_Just_Est_Internacao_F','Pop_Prisional_Med_Seg_Just_Fed_Internacao_M','Pop_Prisional_Med_Seg_Just_Fed_Internacao_F','Pop_Prisional_Med_Seg_Outros_Internacao_M','Pop_Prisional_Med_Seg_Outros_Internacao_F','Pop_Prisional_Med_Seg_Just_Est_Ambulatorial_M','Pop_Prisional_Med_Seg_Just_Est_Ambulatorial_F','Pop_Prisional_Med_Seg_Just_Fed_Ambulatorial_M','Pop_Prisional_Med_Seg_Just_Fed_Ambulatorial_F','Pop_Prisional_Med_Seg_Outros_Ambulatorial_M','Pop_Prisional_Med_Seg_Outros_Ambulatorial_F','Qtde_Regime_Disc_Diferenciado','Sim_Qtde_Prov_Mais_90_Dias_M','Sim_Qtde_Prov_Mais_90_Dias_F','Nao_Qtde_Prov_Mais_90_Dias_M','Nao_Qtde_Prov_Mais_90_Dias_F','Sim_Fechado_Semi_Aguard_Transf_M','Sim_Fechado_Semi_Aguard_Transf_F','Nao_Fechado_Semi_Aguard_Transf_M','Nao_Fechado_Semi_Aguard_Transf_F','Atestado_Pena_Atualizado_Arquivado_M','Atestado_Pena_Atualizado_Arquivado_F','Inclusoes_Originarias_M','Inclusoes_Originarias_F','Alvaras_Soltura_M','Alvaras_Soltura_F','Abandonos_M','Abandonos_F','Obitos_M','Obitos_F','Entrada_Por_Transferencias_Remocoes_M','Entrada_Por_Transferencias_Remocoes_F','Saida_Por_Transferencias_Remocoes_M','Saida_Por_Transferencias_Remocoes_F','Qtde_Saida_Falecimento_Doenca_Parente_Ou_Tratamento_Medico_M','Qtde_Saida_Falecimento_Doenca_Parente_Ou_Tratamento_Medico_F','Qtde_Saida_Semi_Visita_Familiar_M','Qtde_Saida_Semi_Visita_Familiar_F','Qtde_Pop_Pris_18_a_24_M','Qtde_Pop_Pris_18_a_24_F','Qtde_Pop_Pris_25_a_29_M','Qtde_Pop_Pris_25_a_29_F','Qtde_Pop_Pris_30_a_34_M','Qtde_Pop_Pris_30_a_34_F','Qtde_Pop_Pris_35_a_45_M','Qtde_Pop_Pris_35_a_45_F','Qtde_Pop_Pris_46_a_60_M','Qtde_Pop_Pris_46_a_60_F','Qtde_Pop_Pris_61_a_70_M','Qtde_Pop_Pris_61_a_70_F','Qtde_Pop_Pris_acima_70_M','Qtde_Pop_Pris_acima_70_F','Qtde_Pop_Pris_idade_Nao_Informado_M','Qtde_Pop_Pris_idade_Nao_Informado_F','Qtde_Brancos_M','Qtde__Brancos_F','Qtde_Negros_M','Qtde_Negros_F','Qtde_Pardos_M','Qtde_Pardos_F','Qtde_Amarelos_M','Qtde_Amarelos_F','Qtde_Indigenas_M','Qtde_Indigenas_F','Qtde_Outros_Racas_M','Qtde_Outros_Racas_F','Qtde_Nao_Inf_Racas_M','Qtde_Nao_Inf_Racas_F','Qtde_Povo_Indigena1','Qtde_Povo_Indigena2','Qtde_Povo_Indigena3','Qtde_Povo_Indigena4','Qtde_Solteiros_M','Qtde_Solteiros_F','Qtde_Uniao_Estavel_M','Qtde_Uniao_Estavel_F','Qtde_Casados_M','Qtde_Casados_F','Qtde_Separados_M','Qtde_Separados_F','Qtde_Divorciados_M','Qtde_Divorciados_F','Qtde_Viuvo_M','Qtde_Viuvo_F','Qtde_Estado_Civil_Nao_Informado_M','Qtde_Estado_Civil_Nao_Informado_F','Pop_Prisional_PCD_M','Pop_Prisional_PCD_F','Qtde_PCD_Intelectual_M','Qtde_PCD_Intelectual_F','Qtde_PCD_Def_Fisica_M','Qtde_PCD_Def_Fisica_F','Qtde_PCD_Cadeirantes_M','Qtde_PCD_Cadeirantes_F','Qtde_PCD_Def_Auditivo_M','Qtde_PCD_Def_Auditivo_F','Qtde_PCD_Def_Visual_M','Qtde_PCD_Def_Visual_F','Qtde_PCD_Def_Mult_M','Qtde_PCD_Def_Mult_F','Qtde_Analfabeto_M','Qtde_Analfabeto_F','Qtde_Alfabetizados_M','Qtde_Alfabetizados_F','Qtde_Ens_Fund_Incomp_M','Qtde_Ens_Fund_Incomp_F','Qtde_Ens_Fund_Comp_M','Qtde_Ens_Fund_Comp_F','Qtde_Ens_Med_Incomp_M','Qtde_Ens_Med_Incomp_F','Qtde_Ens_Med_Comp_M','Qtde_Ens_Med_Comp_F','Qtde_Ens_Sup_Incomp_M','Qtde_Ens_Sup_Incomp_F','Qtde_Ens_Sup_Comp_M','Qtde_Ens_Sup_Comp_F','Qtde_Ens_Acim_Sup_Comp_M','Qtde_Ens_Acim_Sup_Comp_F','Qtde_Ens_Nao_Inf_M','Qtde_Ens_Nao_Inf_F','Qtde_Certidao_Nasc_M','Qtde_Certidao_Nasc_F','Qtde_RG_M','Qtde_RG_F','Qtde_CPF_M','Qtde_CPF_F','Qtde_Titulo_Eleitor_M','Qtde_Titulo_Eleitor_F','Qtde_Reservista_M','Qtde_Reservista_F','Qtde_CPTS_M','Qtde_CPTS_F','Qtde_Cartao_SUS_M','Qtde_Cartao_SUS_F','Qtde_RNE_M','Qtde_RNE_F','Qtde_Passaporte_Estrangeiros_M','Qtde_Passaporte_Estrangeiros_F','Qtde_Num_Pesooas_Docum_Acima_M','Qtde_Num_Pesooas_Docum_Acima_F','Qtde_Pessoas_Sem_Documento_M','Qtde_Pessoas_Sem_Documento_F','Qtde_Brasileiros_Nato_M','Qtde_Brasileiros_Nato_F','Qtde_Brasileiros_Naturalizados_M','Qtde_Brasileiros_Naturalizados_F','Qtde_Estrangeiros_M','Qtde_Estrangeiros_F','Qtde_Europa_Alemaes_M','Qtde_Europa_Alemaes_F','Qtde_Europa_Austriacos_M','Qtde_Europa_Austriacos_F','Qtde_Europa_Belgas_M','Qtde_Europa_Europa_Belgas_F','Qtde_Europa_Bulgaros_M','Qtde_Europa_Bulgaros_F','Qtde_Europa_Tchecos_M','Qtde_Europa_Tchecos_F','Qtde_Europa_Croatas_M','Qtde_Europa_Croatas_F','Qtde_Europa_Dinamarqueses_M','Qtde_Europa_Dinamarqueses_F','Qtde_Europa_Escoceses_M','Qtde_Europa_Escoceses_F','Qtde_Europa_Espanhois_M','Qtde_Europa_Europa_Espanhois_F','Qtde_Europa_Franceses_M','Qtde_Europa_Franceses_F','Qtde_Europa_Europa_Gragos_M','Qtde_Europa_Gragos_F','Qtde_Europa_Holandeses_M','Qtde_Europa_Holandeses_F','Qtde_Europa_Hungria_M','Qtde_Europa_Hungria_F','Qtde_Europa_Ingleses_M','Qtde_Europa_Ingleses_F','Qtde_Europa_Irlandeses_M','Qtde_Europa_Irlandeses_F','Qtde_Europa_Italianos_M','Qtde_Europa_Italianos_F','Qtde_Europa_Noruega_M','Qtde_Europa_Noruega_F','Qtde_Europa_Galeses_M','Qtde_Europa_Galeses_F','Qtde_Europa_Poloneses_M','Qtde_Europa_Poloneses_F','Qtde_Europa_Portugueses_M','Qtde_Europa_Portugueses_F','Qtde_Europa_Russos_M','Qtde_Europa_Russos_F','Qtde_Europa_Reino_Unido_M','Qtde_Europa_Reino_Unido_F','Qtde_Europa_Romenos_M','Qtde_Europa_Romenos_F','Qtde_Europa_Servos_M','Qtde_Europa_Servos_F','Qtde_Europa_Sueco_M','Qtde_Europa_Sueco_F','Qtde_Europa_Suicos_M','Qtde_Europa_Suicos_F','Qtde_Europa_Outros_Paises_M','Qtde_Europa_Outros_Paises_F','Qtde_Asia_Afeganistao_M','Qtde_Asia_Afeganistao_F','Qtde_Asia_Arabia_Saudita_M','Qtde_Asia_Arabia_Saudita_F','Qtde_Asia_Catar_M','Qtde_Asia_Catar_F','Qtde_Asia_Cazaquistao_M','Qtde_Asia_Cazaquistao_F','Qtde_Asia_China_M','Qtde_Asia_China_F','Qtde_Asia_Coreia_Norte_M','Qtde_Asia_Coreia_Norte_F','Qtde_Asia_Coreia_Sul_M','Qtde_Asia_Coreia_Sul_F','Qtde_Asia_Emirados_Arabes_M','Qtde_Asia_Emirados_Arabes_F','Qtde_Asia_Filipinas_M','Qtde_Asia_Filipinas_F','Qtde_Asia_India_M','Qtde_Asia_India_F','Qtde_Asia_Indonesia_M','Qtde_Asia_Indonesia_F','Qtde_Asia_Ira_M','Qtde_Asia_Ira_F','Qtde_Asia_Iraque_M','Qtde_Asia_Iraque_F','Qtde_Asia_Irael_M','Qtde_Asia_Irael_F','Qtde_Asia_Japao_M','Qtde_Asia_Japao_F','Qtde_Asia_Jordania_M','Qtde_Asia_Jordania_F','Qtde_Asia_Kuwait_M','Qtde_Asia_Kuwait_F','Qtde_Asia_Libano_M','Qtde_Asia_Libano_F','Qtde_Asia_Macau_M','Qtde_Asia_Macau_F','Qtde_Asia_Malasia_M','Qtde_Asia_Malasia_F','Qtde_Asia_Paquistao_M','Qtde_Asia_Paquistao_F','Qtde_Asia_Siria_M','Qtde_Asia_Siria_F','Qtde_Asia_Sri_Lanka_M','Qtde_Asia_Sri_Lanka_F','Qtde_Asia_Tailandia_M','Qtde_Asia_Tailandia_F','Qtde_Asia_Taiwan_M','Qtde_Asia_Taiwan_F','Qtde_Asia_Turquia_M','Qtde_Asia_Turquia_F','Qtde_Asia_Timor_Leste_M','Qtde_Asia_Timor_Leste_F','Qtde_Asia_Vietna_M','Qtde_Asia_Vietna_F','Qtde_Asia_Outros_Paises_M','Qtde_Asia_Outros_Paises_F','Qtde_Africa_Africa_Do_Sul_M','Qtde_Africa_Africa_Do_Sul_F','Qtde_Africa_Angola_M','Qtde_Africa_Angola_F','Qtde_Africa_Argelia_M','Qtde_Africa_Argelia_F','Qtde_Africa_Cabo_Verde_M','Qtde_Africa_Cabo_Verde_F','Qtde_Africa_Camaroes_M','Qtde_Africa_Camaroes_F','Qtde_Africa_Republica_Do_Congo_M','Qtde_Africa_Republica_Do_Congo_F','Qtde_Africa_Costa_Do_Marfim_M','Qtde_Africa_Costa_Do_Marfim_F','Qtde_Africa_Egito_M','Qtde_Africa_Egito_F','Qtde_Africa_Etiopia_M','Qtde_Africa_Etiopia_F','Qtde_Africa_Gana_M','Qtde_Africa_Gana_F','Qtde_Africa_Guine_M','Qtde_Africa_Guine_F','Qtde_Africa_Guine_Bissau_M','Qtde_Africa_Guine_Bissau_F','Qtde_Africa_Libia_M','Qtde_Africa_Libia_F','Qtde_Africa_Madagascar_M','Qtde_Africa_Madagascar_F','Qtde_Africa_Marrocos_M','Qtde_Africa_Marrocos_F','Qtde_Africa_Mocambique_M','Qtde_Africa_Mocambique_F','Qtde_Africa_Nigeria_M','Qtde_Africa_Nigeria_F','Qtde_Africa_Quenia_M','Qtde_Africa_Quenia_F','Qtde_Africa_Ruanda_M','Qtde_Africa_Ruanda_F','Qtde_Africa_Senegal_M','Qtde_Africa_Senegal_F','Qtde_Africa_Serra_Leoa_M','Qtde_Africa_Serra_Leoa_F','Qtde_Africa_Somalia_M','Qtde_Africa_Somalia_F','Qtde_Africa_Tunisia_M','Qtde_Africa_Tunisia_F','Qtde_Africa_Outros_Paises_M','Qtde_Africa_Outros_Paises_F','Qtde_America_Argentina_M','Qtde_America_Argentina_F','Qtde_America_Bolivia_M','Qtde_America_Bolivia_F','Qtde_America_Canada_M','Qtde_America_Canada_F','Qtde_America_Chile_M','Qtde_America_Chile_F','Qtde_America_Colombia_M','Qtde_America_Colombia_F','Qtde_America_Costa_Rica_M','Qtde_America_Costa_Rica_F','Qtde_America_Cuba_M','Qtde_America_Cuba_F','Qtde_America_El_Salvador_M','Qtde_America_El_Salvador_F','Qtde_America_Equador_M','Qtde_America_Equador_F','Qtde_America_EUA_M','Qtde_America_EUA_F','Qtde_America_Quatemala_M','Qtde_America_Quatemala_F','Qtde_America_Quiana_M','Qtde_America_Quiana_F','Qtde_America_Quiana_Francesa_M','Qtde_America_Quiana_Francesa_F','Qtde_America_Haiti_M','Qtde_America_Haiti_F','Qtde_America_Honduras_M','Qtde_America_Honduras_F','Qtde_America_Ilhas_Cayman_M','Qtde_America_Ilhas_Cayman_F','Qtde_America_Jamaica_M','Qtde_America_Jamaica_F','Qtde_America_Mexico_M','Qtde_America_Mexico_F','Qtde_America_Nicaragua_M','Qtde_America_Nicaragua_F','Qtde_America_Panama_M','Qtde_America_Panama_F','Qtde_America_Paraguai_M','Qtde_America_Paraguai_F','Qtde_America_Peru_M','Qtde_America_Peru_F','Qtde_America_Porto_Rico_M','Qtde_America_Porto_Rico_F','Qtde_America_Republica_Dominicana_M','Qtde_America_Republica_Dominicana_F','Qtde_America_Suriname_M','Qtde_America_Suriname_F','Qtde_America_Trinidad_Tobago_M','Qtde_America_Trinidad_Tobago_F','Qtde_America_Uruguai_M','Qtde_America_Uruguai_F','Qtde_America_Venezuela_M','Qtde_America_Venezuela_F','Qtde_America_Outros_Paises_M','Qtde_America_Outros_Paises_F','Qtde_Oceania_Australia_M','Qtde_Oceania_Australia_F','Qtde_Oceania_Nova_Zelandia_M','Qtde_Oceania_Nova_Zelandia_F','Qtde_Oceania_Outros_Paises_M','Qtde_Oceania_Outros_Paises_F','Qtde_Sem_Nacionalidade_M','Qtde_Sem_Nacionalidade_F','Qtde_Filhos_0_6_Meses','Qtde_Filhos_6_Meses_1_Ano','Qtde_Filhos_1_2_Ano','Qtde_Filhos_2_3_Ano','Qtde_Filhos_mais_3_Ano','Qtde_Pop_Prisional_Sem_Filhos_M','Qtde_Pop_Prisional_Sem_Filhos_F','Qtde_Pop_Prisional_1_Filho_M','Qtde_Pop_Prisional_1_Filho_F','Qtde_Pop_Prisional_2_Filhos_M','Qtde_Pop_Prisional_2_Filhos_F','Qtde_Pop_Prisional_3_Filhos_M','Qtde_Pop_Prisional_3_Filhos_F','Qtde_Pop_Prisional_4_Filhos_M','Qtde_Pop_Prisional_4_Filhos_F','Qtde_Pop_Prisional_5_Filhos_M','Qtde_Pop_Prisional_5_Filhos_F','Qtde_Pop_Prisional_6_Filhos_M','Qtde_Pop_Prisional_6_Filhos_F','Qtde_Pop_Prisional_7_Filhos_M','Qtde_Pop_Prisional_7_Filhos_F','Qtde_Pop_Prisional_8_Filhos_M','Qtde_Pop_Prisional_8_Filhos_F','Qtde_Pop_Prisional_9_Filhos_M','Qtde_Pop_Prisional_9_Filhos_F','Qtde_Pop_Prisional_10_Filhos_M','Qtde_Pop_Prisional_10_Filhos_F','Qtde_Pop_Prisional_11_Ou_Mais_Filhos_M','Qtde_Pop_Prisional_11_Ou_Mais_Filhos_F','Qtde_Pop_Prisional_Sem_Inf_Filhos_M','Qtde_Pop_Prisional_Sem_Inf_Filhos_F','Qtde_Pop_Prisional_Com_Visit_Cadast_M','Qtde_Pop_Prisional_Com_Visit_Cadast_F','Qtde_Penas_Ate_6_Meses_M','Qtde_Penas_Ate_6_Meses_F','Qtde_Penas_6_Meses_1_Ano_M','Qtde_Penas_6_Meses_1_Ano_F','Qtde_Penas_1_2_Anos_M','Qtde_Penas_1_2_Anos_F','Qtde_Penas_2_4_Anos_M','Qtde_Penas_2_4_Anos_F','Qtde_Penas_4_8_Anos_M','Qtde_Penas_4_8_Anos_F','Qtde_Penas_8_15_Anos_M','Qtde_Penas_8_15_Anos_F','Qtde_Penas_15_20_Anos_M','Qtde_Penas_15_20_Anos_F','Qtde_Penas_20_30_Anos_M','Qtde_Penas_20_30_Anos_F','Qtde_Penas_30_50_Anos_M','Qtde_Penas_30_50_Anos_F','Qtde_Penas_50_100_Anos_M','Qtde_Penas_50_100_Anos_F','Qtde_Penas_Mais_100_Anos_M','Qtde_Penas_Mais_100_Anos_F','Qtde_Penas_Sem_Inf_Anos_M','Qtde_Penas_Sem_Inf_Anos_F','Qtde_Pop_Prisional_Penas_Remanescentes_0_6_Meses_M','Qtde_Pop_Prisional_Penas_Remanescentes_0_6_Meses_F','Qtde_Pop_Prisional_Penas_Remanescentes_6_Meses_1_Ano_M','Qtde_Pop_Prisional_Penas_Remanescentes_6_Meses_1_Ano_F','Qtde_Pop_Prisional_Penas_Remanescentes_1_2_Anos_M','Qtde_Pop_Prisional_Penas_Remanescentes_1_2_Anos_F','Qtde_Pop_Prisional_Penas_Remanescentes_2_4_Anos_M','Qtde_Pop_Prisional_Penas_Remanescentes_2_4_Anos_F','Qtde_Pop_Prisional_Penas_Remanescentes_4_8_Anos_M','Qtde_Pop_Prisional_Penas_Remanescentes_4_8_Anos_F','Qtde_Pop_Prisional_Penas_Remanescentes_8_15_Anos_M','Qtde_Pop_Prisional_Penas_Remanescentes_8_15_Anos_F','Qtde_Pop_Prisional_Penas_Remanescentes_15_20_Anos_M','Qtde_Pop_Prisional_Penas_Remanescentes_15_20_Anos_F','Qtde_Pop_Prisional_Penas_Remanescentes_20_30_Anos_M','Qtde_Pop_Prisional_Penas_Remanescentes_20_30_Anos_F','Qtde_Pop_Prisional_Penas_Remanescentes_30_50_Anos_M','Qtde_Pop_Prisional_Penas_Remanescentes_30_50_Anos_F','Qtde_Pop_Prisional_Penas_Remanescentes_50_100_Anos_M','Qtde_Pop_Prisional_Penas_Remanescentes_50_100_Anos_F','Qtde_Pop_Prisional_Penas_Remanescentes_Mais_100_Anos_M','Qtde_Pop_Prisional_Penas_Remanescentes_Mais_100_Anos_F','Qtde_Pop_Prisional_Penas_Remanescentes_Sem_Inf_Anos_M','Qtde_Pop_Prisional_Penas_Remanescentes_Sem_Inf_Anos_F','Qtde_Tipo_Penal_Total_M','Qtde_Tipo_Penal_Total_F','Qtde_Homicidios_Simples_M','Qtde_Homicidios_Simples_F','Qtde_Homicidios_Culposo_M','Qtde_Homicidios_Culposo_F','Qtde_Homicidios_Qualificado_M','Qtde_Homicidios_Qualificado_F','Qtde_Aborto_M','Qtde_Aborto_F','Qtde_Lesao_Corporal_M','Qtde_Lesao_Corporal_F','Qtde_Violencia_Domestica_M','Qtde_Violencia_Domestica_F','Qtde_Sequestro_M','Qtde_Sequestro_F','Qtde_Outro_Tipo_Penal_M','Qtde_Outro_Tipo_Penal_F','Qtde_Furto_Simples_M','Qtde_Furto_Simples_F','Qtde_Furto_Qualificado_M','Qtde_Furto_Qualificado_F','Qtde_Roubo_Simples_M','Qtde_Roubo_Simples_F','Qtde_Roubo_Qualificado_M','Qtde_Roubo_Qualificado_F','Qtde_Latrocinio_M','Qtde_Latrocinio_F','Qtde_Extorsao_M','Qtde_Extorsao_F','Qtde_Extorsao_Mediante_Sequestro_M','Qtde_Extorsao_Mediante_Sequestro_F','Qtde_Apropriacao_Indebita_M','Qtde_Apropriacao_Indebita_F','Qtde_Apropriacao_Indebita_Previdenciaria_M','Qtde_Apropriacao_Indebita_Previdenciaria_F','Qtde_Estelionato_M','Qtde_Estelionato_F','Qtde_Receptacao_M','Qtde_Receptacao_F','Qtde_Receptacao_Qualificada_M','Qtde_Receptacao_Qualificada_F','Qtde_Crimes_Ao_Patrimonio_Outros_M','Qtde_Crimes_Ao_Patrimonio_Outros_F','Qtde_Estupro_M','Qtde_Estupro_F','Qtde_Atentado_Violento_Ao_Pudor_M','Qtde_Atentado_Violento_Ao_Pudor_F','Qtde_Estupro_De_Vulneravel_M','Qtde_Estupro_De_Vulneravel_F','Qtde_Corrupcao_Menores_M','Qtde_Corrupcao_Menores_F','Qtde_Trafico_Internacional_Exploracao_Sexual_M','Qtde_Trafico_Internacional_Exploracao_Sexual_F','Qtde_Trafico_Interno_Exploracao_Sexual_M','Qtde_Trafico_Interno_Exploracao_Sexual_F','Qtde_Crimes_Contra_Dignidade_Sexual_Outros_M','Qtde_Crimes_Contra_Dignidade_Sexual_Outros_F','Qtde_Quadrilha_Bando_M','Qtde_Quadrilha_Bando_F','Qtde_Moeda_Falsa_M','Qtde_Moeda_Falsa_F','Qtde_Falsificacao_Doc_Publicos_M','Qtde_Falsificacao_Doc_Publicos_F','Qtde_Falsidade_Ideologica_M','Qtde_Falsidade_Ideologica_F','Qtde_Uso_Doc_Falso_M','Qtde_Uso_Doc_Falso_F','Qtde_Peculato_M','Qtde_Peculato_F','Qtde_Concussao_M','Qtde_Concussao_F','Qtde_Corrupcao_Passiva_M','Qtde_Corrupcao_Passiva_F','Qtde_Corrupcao_Ativa_M','Qtde_Corrupcao_Ativa_F','Qtde_Contrabando_M','Qtde_Contrabando_F','Qtde_Trafico_Drogas_M','Qtde_Trafico_Drogas_F','Qtde_Associacao_Trafico_M','Qtde_Associacao_Trafico_F','Qtde_Trafico_Intern_Drogas_M','Qtde_Trafico_Intern_Drogas_F','Qtde_Porte_Ilegal_Armas_Permitida_M','Qtde_Porte_Ilegal_Armas_Permitida_F','Qtde_Disparo_Arma_Fogo_M','Qtde_Disparo_Arma_Fogo_F','Qtde_Porte_Ilegal_Armas_Restrita_M','Qtde_Porte_Ilegal_Armas_Restrita_F','Qtde_Comercio_Ilegal_Armas_M','Qtde_Comercio_Ilegal_Armas_F','Qtde_Trafico_Internacional_Armas_M','Qtde_Trafico_Internacional_Armas_F','Qtde_Homic_Culposo_Condu_Veiculo_Automotor_M','Qtde_Homic_Culposo_Condu_Veiculo_Automotor_F','Qtde_Cirmes_Transito_Outros_M','Qtde_Cirmes_Transito_Outros_F','Qtde_Estatuto_Crianca_Adolesc_M','Qtde_Estatuto_Crianca_Adolesc_F','Qtde_Genocidio_M','Qtde_Genocidio_F','Qtde_Crimes_Tortura_M','Qtde_Crimes_Tortura_F','Qtde_Crimes_Contra_Meio_Amb_M','Qtde_Crimes_Contra_Meio_Amb_F','Qtde_Tipificacao_Criminal_M','Qtde_Tipificacao_Criminal_F','Qtde_Sem_Tipificacao_Criminal_M','Qtde_Sem_Tipificacao_Criminal_F','Qtde_Trab_Externo_Setor_Primario_M','Qtde_Trab_Externo_Setor_Primario_F','Qtde_Trab_Interno_Setor_Primario_M','Qtde_Trab_Interno_Setor_Primario_F','Qtde_Trab_Externo_Setor_Secundario_M','Qtde_Trab_Externo_Setor_Secundario_F','Qtde_Trab_Interno_Setor_Secundario_M','Qtde_Trab_Interno_Setor_Secundario_F','Qtde_Trab_Externo_Setor_Terceario_M','Qtde_Trab_Externo_Setor_Terceario_F','Qtde_Trab_Interno_Setor_Terceario_M','Qtde_Trab_Interno_Setor_Terceario_F','Qtde_Trab_Apoio_Estb_Interno_M','Qtde_Trab_Apoio_Estb_Interno_F','Qtde_Trab_Externo_Parc_Privada_Setor_Primario_M','Qtde_Trab_Externo_Parc_Privada_Setor_Primario_F','Qtde_Trab_Interno_Parc_Privada_Setor_Primario_M','Qtde_Trab_Interno_Parc_Privada_Setor_Primario_F','Qtde_Trab_Externo_Parc_Privada_Setor_Secundario_M','Qtde_Trab_Externo_Parc_Privada_Setor_Secundario_F','Qtde_Trab_Interno_Parc_Privada_Setor_Secundario_M','Qtde_Trab_Interno_Parc_Privada_Setor_Secundario_F','Qtde_Trab_Externo_Parc_Privada_Setor_Terceario_M','Qtde_Trab_Externo_Parc_Privada_Setor_Terceario_F','Qtde_Trab_Interno_Parc_Privada_Setor_Terceario_M','Qtde_Trab_Interno_Parc_Privada_Setor_Terceario_F','Qtde_Trab_Externo_Parc_Publico_Setor_Primario_M','Qtde_Trab_Externo_Parc_Publico_Setor_Primario_F','Qtde_Trab_Interno_Parc_Publico_Setor_Primario_M','Qtde_Trab_Interno_Parc_Publico_Setor_Primario_F','Qtde_Trab_Externo_Parc_Publico_Setor_Secundario_M','Qtde_Trab_Externo_Parc_Publico_Setor_Secundario_F','Qtde_Trab_Interno_Parc_Publico_Setor_Secundario_M','Qtde_Trab_Interno_Parc_Publico_Setor_Secundario_F','Qtde_Trab_Externo_Parc_Publico_Setor_Terceario_M','Qtde_Trab_Externo_Parc_Publico_Setor_Terceario_F','Qtde_Trab_Interno_Parc_Publico_Setor_Terceario_M','Qtde_Trab_Interno_Parc_Publico_Setor_Terceario_F','Qtde_Trab_Externo_Sem_Fim_Luc_Setor_Primario_M','Qtde_Trab_Externo_Sem_Fim_Luc_Setor_Primario_F','Qtde_Trab_Interno_Sem_Fim_Luc_Setor_Primario_M','Qtde_Trab_Interno_Sem_Fim_Luc_Setor_Primario_F','Qtde_Trab_Externo_Sem_Fim_Luc_Setor_Secundario_M','Qtde_Trab_Externo_Sem_Fim_Luc_Setor_Secundario_F','Qtde_Trab_Interno_Sem_Fim_Luc_Setor_Secundario_','Qtde_Trab_Interno_Sem_Fim_Luc_Setor_Secundario_F','Qtde_Trab_Externo_Sem_Fim_Luc_Setor_Terceario_M','Qtde_Trab_Externo_Sem_Fim_Luc_Setor_Terceario_F','Qtde_Trab_Interno_Sem_Fim_Luc_Setor_Terceario_M','Qtde_Trab_Interno_Sem_Fim_Luc_Setor_Terceario_F','Qtde_Nao_Tem_Remuneracao_M','Qtde_Nao_Tem_Remuneracao_F','Qtde_Remuneracao_3_4_Sal_Min_M','Qtde_Remuneracao_3_4_Sal_Min_F','Qtde_Remuneracao_3_4_Mais_1_Sal_Min_F','Qtde_Remuneracao_1_2_Sal_Min_M','Qtde_Remuneracao_1_2_Sal_Min_F','Qtde_Remuneracao_Mais_2_Sal_Min_M','Qtde_Remuneracao_Mais_2_Sal_Min_F','Qtde_Remuneracao_Sem_Inf_M','Qtde_Remuneracao_Sem_Inf_F','Qtde_Presencial_Atividades_Educacionais_M','Qtde_Presencial_Atividades_Educacionais_F','Qtde_EAD_Atividades_Educacionais_M','Qtde_EAD_Atividades_Educacionais_F','Qtde_Certif_Atividade_Educacional_Semestre_M','Qtde_Certif_Atividade_Educacional_Semestre_F','Qtde_Presencial_Alfabetizacao_M','Qtde_Presencial_Alfabetizacao_F','Qtde_EAD_Alfabetizacao_M','Qtde_EAD_Alfabetizacao_F','Qtde_Certif_Alfabetizacao_Semestre_M','Qtde_Certif_Alfabetizacao_Semestre_F','Qtde_Presencial_Ensino_Fundamental_M','Qtde_Presencial_Ensino_Fundamental_F','Qtde_EAD_Ensino_Fundamental_M','Qtde_EAD_Ensino_Fundamental_F','Qtde_Certif_Ensino_Fundamental_Semestre_M','Qtde_Certif_Ensino_Fundamental_Semestre_F','Qtde_Presencial_Ensino_Medio_M','Qtde_Presencial_Ensino_Medio_F','Qtde_EAD_Ensino_Medio_M','Qtde_EAD_Ensino_Medio_F','Qtde_Certif_Ensino_Medio_Semestre_M','Qtde_Certif_Ensino_Medio_Semestre_F','Qtde_Presencial_Ensino_Superior_M','Qtde_Presencial_Ensino_Superior_F','Qtde_EAD_Ensino_Superior_M','Qtde_EAD_Ensino_Superior_F','Qtde_Certif_Ensino_Superior_Semestre_M','Qtde_Certif_Ensino_Superior_Semestre_F','Qtde_Presencial_Curso_Tec_800h_M','Qtde_Presencial_Curso_Tec_800h_F','Qtde_EAD_Curso_Tec_800h_M','Qtde_EAD_Curso_Tec_800h_F','Qtde_Certif_Semestre_Curso_Tec_800h_M','Qtde_Certif_Semestre_Curso_Tec_800h_F','Qtde_Presencial_Capacitacao_Profissional_M','Qtde_Presencial_Capacitacao_Profissional_F','Qtde_EAD_Capacitacao_Profissional_M','Qtde_EAD_Capacitacao_Profissional_F','Qtde_Certif_Capacitacao_Profissional_Semestre_M','Qtde_Certif_Capacitacao_Profissional_Semestre_F','Qtde_Presencial_Remicao_Leitura_M','Qtde_Presencial_Remicao_Leitura_F','Qtde_EAD_Remicao_Leitura_M','Qtde_EAD_Remicao_Leitura_F','Qtde_Certif_Remicao_Leitura_Semestre_M','Qtde_Certif_Remicao_Leitura_Semestre_F','Qtde_Presencial_Remicao_Esporte_Semestre_M','Qtde_Presencial_Remicao_Esporte_Semestre_F','Qtde_EAD_Remicao_Esporte_Semestre_M','Qtde_EAD_Remicao_Esporte_Semestre_F','Qtde_Certif_Remicao_Esporte_Semestre_M','Qtde_Certif_Remicao_Esporte_Semestre_F','Qtde_Presencial_Remicao_Atividades_Complem_M','Qtde_Presencial_Remicao_Atividades_Complem_F','Qtde_EAD_Remicao_Atividades_Complem_M','Qtde_EAD_Remicao_Atividades_Complem_F','Qtde_Certif_Atividades_Complem_Semestre_M','Qtde_Certif_Atividades_Complem_Semestre_F','Qtde_Trabalham_E_Estudam_M','Qtde_Trabalham_E_Estudam_F','Qtde_Familias_Receb_Aux_Exclusao_M','Qtde_Consult_Medicas_Externas_M','Qtde_Consult_Medicas_Externas_F','Qtde_Consult_Medicas_Internas_M','Qtde_Consult_Piscologicas_F','Qtde_Consult_Odontologicas_M','Qtde_Consult_Odontologicas_F','Qtde_Exames_M','Qtde_Exames_F','Qtde_Inter_Cirurgicas_M','Qtde_Inter_Cirurgicas_F','Qtde_Vacinas_M','Qtde_Vacinas_F','Qtde_HIV_Positivo_M','Qtde_HIV_Positivo_F','Qtde_Sifilis_Positivo_M','Qtde_Sifilis_Positivo_F','Qtde_Hepatite_Positivo_M','Qtde_Hepatite_Positivo_F','Qtde_Tuberculose_Positivo_M','Qtde_Tuberculose_Positivo_F','Qtde_DST_Outros_M','Qtde_DST_Outros_F','Qtde_Obitos_Naturais_M','Qtde_Obitos_Naturais_F','Qtde_Obitos_Criminais_M','Qtde_Obitos_Criminais_F','Qtde_Obitos_Suicidios_M','Qtde_Obitos_Suicidios_F','Qtde_Obitos_Acidentais_M','Qtde_Obitos_Acidentais_F','Qtde_Obitos_Causa_Desconhecidas_M','Qtde_Obitos_Causa_Desconhecidas_F','Qtde_Visitas_Registradas_Semestre','Qtde_Receberam_Visitas_Semestre_M','Qtde_Receberam_Visitas_Semestre_F','Inspecao_No_Semestre_CNPCP','Inspecao_No_Semestre_Conselho_Penintenciario','Inspecao_No_Semestre_Conselho_Comunidade','Inspecao_No_Semestre_Ouvidoria_Sistem_Prisional','Inspecao_No_Semestre_Defensoria_Publica','Inspecao_No_Semestre_Judiciario','Inspecao_No_Semestre_Ministerio_Publico','Inspecao_No_Semestre_Outros']].copy()
X = df_2014_new  #colunas independentes

y = df_2014.iloc[:,1345].astype(int)    #Coluna Target SelectKBest para extrair top melhores features

bestfeatures = SelectKBest(score_func=chi2, k=20)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#Concatenando os 2 Dataframes para melhor visualização

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #Nomeando as colunas do dataframe

print(featureScores.nlargest(20,'Score'))  #Mostrando as 0 melhores features
# Treinando um modelo de RandomForest com os dados

model = RandomForestClassifier(max_leaf_nodes=50)

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
df_us = df_2014[['Qtde_Visitas_Registradas_Semestre','Qtde_Receberam_Visitas_Semestre_M','Qtde_Saida_Semi_Visita_Familiar_M','Qtde_Tipo_Penal_Total_M','Qtde_Pop_Prisional_11_Ou_Mais_Filhos_M','Qtde_Vacinas_M','Qtde_Receberam_Visitas_Semestre_F','Qtde_Pop_Prisional_Penas_Remanescentes_2_4_Anos_M','Qtde_Consult_Medicas_Internas_M','Qtde_Regime_Disc_Diferenciado','Qtde_Roubo_Qualificado_M','Qtde_Tipificacao_Criminal_M','Qtde_Penas_Sem_Inf_Anos_M','Qtde_Estelionato_F','Qtde_Pessoas_Sem_Documento_M','Inclusoes_Originarias_M','Qtde_Pop_Prisional_Sem_Inf_Filhos_M','Qtde_Pop_Prisional_Com_Visit_Cadast_M']]
corrmat = df_us.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

g=sns.heatmap(df_us[top_corr_features].corr(),annot=True,cmap="RdYlGn")
df_2014_new['target'] = df_2014['Capacidade_Pop_Prisional'].astype(int)
function = '''

target ~ Qtde_Visitas_Registradas_Semestre

+Qtde_Receberam_Visitas_Semestre_M

+Qtde_Saida_Semi_Visita_Familiar_M

+Qtde_Receberam_Visitas_Semestre_F

+Qtde_Tipo_Penal_Total_M

+Qtde_Pop_Prisional_11_Ou_Mais_Filhos_M

+Qtde_Vacinas_M

+Qtde_Consult_Medicas_Internas_M

+Pop_Prisional_Semi_Just_Est_F

+Inclusoes_Originarias_M

+Pop_Prisional_Fechado_Outros_M

+Qtde_Pop_Prisional_Penas_Remanescentes_2_4_Anos_M

+Qtde_Regime_Disc_Diferenciado

+Pop_Prisional_Fechado_Just_Est_M

+Qtde_Pop_Prisional_Penas_Remanescentes_Sem_Inf_Anos_M

+Qtde_Brasileiros_Naturalizados_F

+Pop_Prisional_Prov_Just_Est_M

+Pop_Prisional_Fechado_Just_Est_F

+Qtde_Roubo_Qualificado_M

+Saida_Por_Transferencias_Remocoes_M

'''

results = smf.ols(function, data=df_2014_new).fit()

print(results.summary())
df_fi = df_2014[['Qtde_Pop_Pris_61_a_70_M', 'Qtde_Solteiros_M','Qtde_Brasileiros_Nato_M','Qtde_Uniao_Estavel_M','Qtde_Ens_Med_Incomp_M','Qtde_Ens_Fund_Incomp_M','Qtde_Brancos_M','Qtde_Pop_Pris_18_a_24_M','Qtde_Ens_Fund_Comp_M','Qtde_Negros_M']]
corrmat = df_fi.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

g=sns.heatmap(df_fi[top_corr_features].corr(),annot=True,cmap="RdYlGn")
function = '''

target ~ Qtde_Pop_Pris_61_a_70_M

+Qtde_Solteiros_M

+Qtde_Brasileiros_Nato_M

+Qtde_Uniao_Estavel_M

+Qtde_Ens_Med_Incomp_M

+Qtde_Ens_Fund_Incomp_M

+Qtde_Brancos_M

+Qtde_Pop_Pris_18_a_24_M

+Qtde_Ens_Fund_Comp_M

+Qtde_Negros_M

'''

results = smf.ols(function, data=df_2014_new).fit()

print(results.summary())
df_rnd = df_2014[['Qtde_Negros_M', 'Qtde_Brasileiros_Nato_M','Qtde_Pop_Pris_46_a_60_M','Qtde_Solteiros_M','Qtde_Pop_Pris_35_a_45_M','Qtde_Analfabeto_M','Qtde_Pop_Pris_18_a_24_M','Qtde_Ens_Med_Comp_M','Inspecao_No_Semestre_Ministerio_Publico','Qtde_Pardos_M']]
corrmat = df_rnd.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

g=sns.heatmap(df_rnd[top_corr_features].corr(),annot=True,cmap="RdYlGn")
function = '''

target ~ Qtde_Negros_M

+Qtde_Brasileiros_Nato_M

+Qtde_Pop_Pris_46_a_60_M

+Qtde_Solteiros_M

+Qtde_Pop_Pris_35_a_45_M

+Qtde_Analfabeto_M

+Qtde_Pop_Pris_18_a_24_M

+Qtde_Ens_Med_Comp_M

+Inspecao_No_Semestre_Ministerio_Publico

+Qtde_Pardos_M

'''

results = smf.ols(function, data=df_2014_new).fit()

print(results.summary())
df_2014_new = df_2014_new.loc[:,~df_2014_new.columns.duplicated()]
X = df_2014_new.drop(['target'], axis=1)

y = df_2014.iloc[:,1345].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
d_train = xgboost.DMatrix(X_train, label=y_train)

d_test = xgboost.DMatrix(X_test, label=y_test)
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42)



dtr = DecisionTreeClassifier(max_depth=3)

dtr.fit(X_train, y_train)

yhat = dtr.predict(X_test)

yhat_proba = 1-dtr.predict_proba(X_test)[:, :1]

yhat_proba_train = 1-dtr.predict_proba(X_train)[:, :1]



precision = metrics.precision_score(y_test, yhat, average='micro')

recall = metrics.recall_score(y_test, yhat, average='micro')

auc = accuracy_score(y_test, yhat, )

#auc = metrics.roc_auc_score(y_test, yhat_proba, average='weighted')



print(

    f'Precisao: {round(precision,4)}, Recall:{round(recall,4)}, AUC:{round(auc,4)}')
sc = StandardScaler() 



sc.fit(X_train)

  

X_train_sc = sc.transform(X_train) 

X_test_sc = sc.transform(X_test)
pca = PCA(n_components = 4)

pca.fit(X_test_sc)
modelo_MQO = LinearRegression()

modelo_RF = RandomForestRegressor()
modelo_MQO.fit(X_train, y_train)
modelo_RF.fit(X_train, y_train)
from sklearn.cluster import KMeans

wcss = []



for i in range(1, 11):

    kmeans= KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

    

#Plotting The elbow'



plt.plot(range(1, 11), wcss)

plt.title('The elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS') #within cluster sum of squares

plt.show()
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
print(pca.components_)
print(pca.explained_variance_)
print(np.round(pca.explained_variance_ratio_, 4))
pd.DataFrame((np.round(pca.components_, 4)), columns= df_2014_new.drop(['target'], axis=1).columns).T
pca = PCA(n_components=None)

pca.fit(X_train_sc)
principalComponents = pca.fit_transform(X_test_sc)

PCA_components = pd.DataFrame(principalComponents)

print (*principalComponents)
print(np.round(pca.explained_variance_ratio_,3))
np.cumsum(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100)

plt.xlabel('número de componentes')

plt.ylabel('Variância acumulada explicativa')
res = pca.transform(X_train_sc)

index_name = ['PCA_'+str(k)for k in range (0,len(res))]

res
print (df_2014_new.shape)

print (res.shape)
sklearn_pca = PCA(n_components = 2)

Y_sklearn = sklearn_pca.fit_transform(res)

kmeans = KMeans(n_clusters=9, max_iter=6000, algorithm = 'auto')

fitted = kmeans.fit(Y_sklearn)

prediction = kmeans.predict(Y_sklearn)
ks = range(1, 10)

inertias = [] 

for k in ks:

    # Create a KMeans instance with k clusters: model

    model = KMeans(n_clusters=k)

    

    # Fit model to samples

    model.fit(PCA_components.iloc[:,:3])

    

    # Append the inertia to the list of inertias

    inertias.append(model.inertia_)

    

plt.plot(ks, inertias, '-o', color='blue')

plt.xlabel('number of clusters, k')

plt.ylabel('inertia')

plt.xticks(ks)

plt.show()
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
logit.fit(X_train,y_train)
print("Acuracia:", logit.score(X_test, y_test))
X_test.values[0]
logit.predict(np.array(X_test.values[0]).reshape(1,-1))
X_test.values[10]
logit.predict(np.array(X_test.values[10]).reshape(1,-1))
rndForest = RandomForestClassifier(max_depth=10, random_state=17, n_estimators=100)

rndForest.fit(X_train,y_train)
y_pred = rndForest.predict(X_test)

print(rndForest.__class__.__name__, accuracy_score(y_test,y_pred)) 
X = df_2014_new.drop(['target'], axis=1)

y = df_2014.iloc[:,1345].astype(int)
X_train_ = X_train.sample(frac=0.10, replace=True, random_state=1)

X_train_.shape
y_train_ = y_train.sample(frac=0.10, replace=True, random_state=1)

y_train_.shape
# Create XGB Classifier object

xgb_clf = xgboost.XGBClassifier(eval_metric=[

                                "merror", "map", "auc"], objective="binary:logistic")



# Altere os parametros para tree_method = "gpu_exact" e predictor = "gpu_predictor" para rodar o modelo na GPU (precisa instalar CUDA)



# Create parameter grid

parameters = {"learning_rate": [0.1, 0.01, 0.001],

              "gamma": [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],

              "max_depth": [2, 4, 7, 10],

              "colsample_bytree": [0.3, 0.6, 0.8, 1.0],

              "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],

              "reg_alpha": [0, 0.5, 1],

              "reg_lambda": [1, 1.5, 2, 3, 4.5],

              "min_child_weight": [1, 3, 5, 7],

              "n_estimators": [100]}

# Create RandomizedSearchCV Object

xgb_rscv = RandomizedSearchCV(xgb_clf, param_distributions=parameters,

                              scoring="f1_micro", cv=3,

                              random_state=42, verbose=True, n_jobs=-1)



# Fit the model

model_xgboost = xgb_rscv.fit(X_train_, y_train_)
xgb_rscv.best_params_
# params = {

#     "eta": 0.01,

#     "objective": "binary:logistic",

#     "subsample": 0.5,

#     "base_score": np.mean(y_train),

#     "eval_metric": "logloss"

# }



# # or 



params = model_xgboost.best_params_
model = xgboost.train(params, d_train, 1500, evals=[

                      (d_test, "test")], verbose_eval=25, early_stopping_rounds=10)
# this takes a minute or two since we are explaining over 30 thousand samples in a model with over a thousand trees

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)
shap.force_plot(explainer.expected_value, shap_values[10,:], X_train_.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values[:100,:], X_train_.iloc[:100,:])
shap.summary_plot(shap_values, X_train_, plot_type="bar")
shap.summary_plot(shap_values, X)