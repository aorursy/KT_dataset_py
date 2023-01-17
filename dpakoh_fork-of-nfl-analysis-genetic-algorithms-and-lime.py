import pickle

import pandas as pd

from zipfile import ZipFile

from google.colab import drive



#Reading the sources from Google Drive

drive.mount('/content/drive',force_remount=True)



#Because the sources are big enough, the program crashes when the .csv files ares being read, therefore 

#a workaround was created.

zip_file = ZipFile('/content/drive/My Drive/Special_Analysis/nfl-playing-surface-analytics.zip')



#Reading all the files inside the .zip container.

df = pd.read_csv(zip_file.open("PlayerTrackData.csv"))

df2 = pd.read_csv(zip_file.open("PlayList.csv"))

df3 = pd.read_csv(zip_file.open("InjuryRecord.csv"))
#Printing a piece of the data just to make sure of the correct lecture.

print("Player Track Data")

print(df.head())



print("PlayList")

print(df2.head())



print("Injury Record")

print(df3.head())
import pandas as pd

import pickle



#The next step consists on creating a training dataset; based on the information retrieved on the website, supposedly all

#three dataframes have in common the field "PlayKey", so we merge them and export the corresponding result into a

#pickle file.



#print(df.groupby(['PlayKey']).count())

#print(df2.groupby(['PlayKey']).count())

#print(df3.groupby(['PlayKey']).count())



df_casi = pd.merge(df2,df3, how='inner', on = 'PlayKey')



df_final = pd.merge(df_casi, df, how='inner', on='PlayKey')#.to_pickle("/content/drive/My Drive/Special_Analysis/df_final.pickle")
def suma(row):

    return row["DM_M1"] + row["DM_M7"]*2 + (row["DM_M28"]*2**2) +  (row["DM_M42"]*(2**3)) 			



#Applying the suma function to all players.

df_final["SumResults"] = df_final.apply (lambda row: suma(row), axis=1)
#Converting dummy variables and saving the final dataset.

columns_dummies = [ 

                    'RosterPosition',

                    'StadiumType',

                    'FieldType',

                    'Weather',

                    'PlayType',

                    'Position',

                    'PositionGroup',

                    'Surface',

                    'event',

                    #'BodyPart'

                   ]



df_final_2 = pd.get_dummies(df_final, columns=columns_dummies)

print(df_final_2.columns)



#Shuffling and saving results in the final pickle.

df_final_2 = df_final_2.sample(frac=1).reset_index(drop=True)

df_final_2.to_pickle("/content/drive/My Drive/Special_Analysis/df_final_2.pickle")
#Reading the final dataset in order to check if everything is in order.

df_final_1 = pd.read_pickle('/content/drive/My Drive/Special_Analysis/df_final_2.pickle') 

df_final_1.head()
#First of all we need to load some packages.



import numpy as np

import pandas as pd

import random as aleatorio 

import operator as operador 



from sklearn import preprocessing

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor



from sklearn.svm import SVC,LinearSVC

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss,roc_auc_score

from sklearn.model_selection import train_test_split
df_final[df_final['BodyPart'] == 'Foot'].groupby(['SumResults']).count()
df_final[df_final['BodyPart'] == 'Knee'].groupby(['SumResults']).count()
df_final[df_final['BodyPart'] == 'Ankle'].groupby(['SumResults']).count()
#Creating target with the features defined above

def target(row):

    flag = 0

    if row["SumResults"] == 1 or row["SumResults"] == 3 or (row["SumResults"] == 7 and row["BodyPart"] == "Foot"):

       flag = 1



    return flag



#Applying the target function to all players.

df_final["Target"] = df_final.apply (lambda row: target(row), axis=1)



#Counting results.

print(df_final.groupby(['Target']).count())
class Individual:

      """

         | La base de toda operación lógica.

         | Consiste en una abstracción de un elemento simple en función de un ecosistema.

          Si bien la parte esencial es el cromosoma, en esta implementación se añaden algunos elementos extra

          con la finalidad de facilitar ciertas operaciones.



         :param complete_chromosome: El cromosoma que conformará al Individuo.

         

         :type complete_chromosome: Array

         

         :returns: Individual

         :rtype: Instance

      """



             

      #dummy columns viene desde la transformación del set, antes

      def __init__(self,

                   complete_chromosome,

                   complete_chromosome_length,

                   predictive_variable,

                   training_set,

                   testing_set,

                   number_of_columns,

                   available_global_columns,

                   forbidden_columns

                   ):

          #Se almacenan los atributos para poder usarlos posteriormente.

          self.__decision_variables = [number_of_columns] 

          self.__complete_chromosome = complete_chromosome

          self.__complete_chromosome_length = complete_chromosome_length

          #self.__model = LinearRegression()

          #self.__model = LogisticRegression(solver = 'liblinear')

          #self.__model = LinearSVC(random_state=0, tol=1e-5)

          #self.__model = LinearSVC(random_state = 0, tol = 1e-5,max_iter = 10000)

          #self.__model = SVC(decision_function_shape='ovo',gamma="auto")

          self.__model = ElasticNet(alpha=0.1, l1_ratio=0.7)

          self.__predictive_variable = predictive_variable

          self.__training_set = training_set

          self.__testing_set = testing_set

          self.__available_global_columns = available_global_columns

          self.__forbidden_columns = forbidden_columns

          self.__query = [] 

		  

          

          #Aquí se almacenarán las funciones objetivo ya evaluadas.

          #Cabe mencionar que éstas son:

          #Número de variables en el modelo i.e. número de columnas en el conjunto de entrenamiento.

          #AUROC 

          #LOGLOSS

          #

          #Salvo en el caso del AUROC que se busca maximizar (minimizando el mínimo),

          #en los demás se opta por la minimización.

          self.__evaluated_functions = [number_of_columns,0,0]



          #Los siguientes atributos auxilian en las comparaciones hechas para poder encontrar al

          #mejor Individuo (véase la clase Community).

          self.__dominates = 0

          self.__is_dominated = 0

          self.__rank = 0

          self.__fitness = 0.0

          self.__niche_count = 0.0



          #La siguiente variable se utiliza sobre todo en técnicas de Selección.

          self.__expected_value = 0.0



          self.__generate_query() 





      def get_complete_chromosome(self):

          """

             Regresa el cromosoma del Individuo.

 

             :returns: El cromosoma.

             :rtype: Array

          """



          #Regresa el valor de la variable asociada al cromosoma.

          return self.__complete_chromosome

      

	  

      def get_decision_variables(self):

          return self.__decision_variables





      def get_query(self):

	      return self.__query



		  

      def get_evaluated_functions(self):

          """

             Regresa las funciones objetivo evaluadas.

 

             :returns: Las funciones objetivo evaluadas.

             :rtype: List

          """



          #Se regresa el valor de esta variable.

          return self.__evaluated_functions

      

	  

      def get_expected_value(self):

	      return self.__expected_value





      def get_pareto_dominates(self):

          """

             Regresa el número de soluciones que son dominadas por 

             el actual Individuo.

 

             :returns: El número de soluciones dominadas.

             :rtype: Integer

          """



          #Se regresa este atributo.

          return self.__dominates





      def set_pareto_dominates(self,value):

          """

             Actualiza el número de soluciones dominadas por el

             Individuo actual.

 

             :param value: El valor a actualizar.

             :type value: Integer

          """



          #Se actualiza al valor correspondiente.

          self.__dominates = value     





      def get_pareto_dominated(self):

          """

             Regresa el número de soluciones que dominan al 

             Individuo actual.

 

             :returns: El número de soluciones que dominan a la actual.

             :rtype: Integer

          """



          #Se regresa el valor

          return self.__is_dominated





      def set_pareto_dominated(self,value):

          """

             Actualiza el número de soluciones que dominan a la

             solución actual.

 

             :param value: El valor a actualizar.

             :type value: Integer

          """



          #Se actualiza el valor.

          self.__is_dominated = value     



      

      def get_rank(self):

          """

             Regresa la puntuación **(rank)** que se le designó al Individuo

             **(véase la clase Community)**.

 

             :returns: El rango.

             :rtype: Float

          """



          #Se regresa el rango.

          return self.__rank





      def set_rank(self,rank):

          """

             Actualiza el rango del Individuo.

 

             :param rank: El valor a actualizar.

             :type rank: Float

          """



          #Se actualiza al valor actual.

          self.__rank = rank





      def get_fitness(self):

          """

             Regresa el Fitness del Individuo.

 

             :returns: El Fitness.

             :rtype: Float

          """



          #Se regresa el valor.

          return self.__fitness





      def set_fitness(self,value):

          """

             Actualiza el valor del Fitness.

 

             :param value: El valor a actualizar.

             :type value: Float

          """



          #Se actualiza el valor.

          self.__fitness = value     



     

      def get_niche_count(self):

          """

             Regresa el valor niche para el Individuo.

 

             :returns: El tamaño niche.

             :rtype: Float

          """



          #Se regresa el valor.

          return self.__niche_count

 



      def set_niche_count(self,value):

          """

             Actualiza el valor niche.

 

             :param value: El valor a actualizar.

             :type value: Float

          """



          #Se actualiza el valor

          self.__niche_count = value





      def set_expected_value(self,value):

          """

             Actualiza el valor esperado del Individuo.

 

             :param value: El valor a actualizar.

             :type value: Float

          """



          #Se actualiza el valor.

          self.__expected_value = value

  



      def evaluate_functions(self):

          self.__model.fit(self.__training_set[self.__query], self.__training_set[self.__predictive_variable])

          result = self.__model.predict(self.__testing_set[self.__query])

          self.__evaluated_functions[1] = -1*roc_auc_score(self.__testing_set[self.__predictive_variable],result)

          self.__evaluated_functions[2] = log_loss(self.__testing_set[self.__predictive_variable],result)

          

          

      def __generate_query(self):

          for x in range (self.__complete_chromosome_length):

              current_gene = self.__complete_chromosome[x]

              if current_gene == '1':

                 current_name = self.__available_global_columns[x]

				 

                 if current_name not in self.__forbidden_columns:

                    self.__query.append(current_name)

                 



      def print_info(self):

          """

             Imprime las características básicas del Individuo **(en consola)**.

          """

          print("    Complete chromosome: " + str(self.__complete_chromosome))

          print("    Decision variables: " + str(self.__decision_variables))

          print("    Evaluated functions (same position than Vector Functions): " + str(self.__evaluated_functions))

          print("    Rank:    " + str(self.__rank) + ". Fitness: " + str(self.__fitness) + ". Expected value: " + str(self.__expected_value))

          print("    Pareto dominates: " + str(self.__dominates) + ". Pareto is dominated: " + str(self.__is_dominated) + ". Niche count: " + str(self.__niche_count))

          print("\n")





#*************************************************************************************************************************************



 

class Population:

      """

         Consiste en un conjunto de instancias de la clase Individual, proporcionando además métodos y atributos que 

         se manifiestan tanto en grupo como de manera individual.



         :param population_size: El tamaño de la población.

         :param vector_variables: Lista con las variables de decisión y sus rangos.

         :param available_expressions: Diccionario que contiene algunas funciones escritas como azúcar sintáctica

                                       para que puedan ser utilizadas más fácilmente por el usuario y evaluadas

                                       más ŕapidamente en el programa **(véase Controller/XML/PythonExpressions.xml)**.

         

         :type population_size: Integer

         :type vector_variables: List

         

         :returns: Population

         :rtype: Instance

      """





      def __init__(

               	   self,

                   population_size,

				   vector_variables,

				   predictive_variable,

                   training_set,

                   testing_set,

				   available_global_columns,

				   forbidden_columns

                  ):



          #Se almacenan los valores para que puedan ser usado posteriormente, a veces por clases externas .

          #Los siguientes atributos indican características básicas que tendrán los individuos.

          self.__population_size = population_size

          self.__vector_variables = vector_variables 

          self.__predictive_variable = predictive_variable

          self.__training_set = training_set

          self.__testing_set = testing_set

          self.__available_global_columns = available_global_columns

          self.__forbidden_columns = forbidden_columns

		  

          #La siguiente estructura almacenará a los individuos, que no es otra cosa que un arreglo de tamaño

          #fijo.

          self.__population = [0.0]*self.__population_size  



          #las siguientes variables contendrán números que se calcularán con métodos grupales

          self.__total_fitness = 0.0

          self.__total_expected_value = 0.0



          #Esta estructura almacena los valores mínimo y máximo respectivamente de cada 

          #variable de decisión. Se necesitan para poder obtener el valor sigma share.

          self.__decision_variables_extreme_values = [[0,0]]



          #Esta estructura almacena los valores mínimo y máximo respectivamente de cada 

          #función objetivo. Se necesitan para poder obtener el valor sigma share.

          #Esl 3 es por cada función objetivo que se quiere calcular

          self.__objective_functions_extreme_values = [[0,0]]*3





      def get_individuals(self):

          """

             Regresa los individuos de la Población.

 

             :returns: Estructura que contiene a los Individuos de la Población.

             :rtype: Array

          """



          #Se regresa la estructura.

          return self.__population





      def get_size(self):

          """

             Otorga el tamaño de la Población.

 

             :returns: El tamaño de la Población.

             :rtype: Integer

          """

         

          #Se regresa el atributo concerniente al tamaño de la población.

          return self.__population_size

          



      def get_vector_variables(self):

          """

             Regresa el vector de variables de decisión.

 

             :returns: Conjunto que contiene las variables de decisión con sus rangos.

             :rtype: List

          """



          #Se obtiene el atributo relativo a las variables de decisión.

          return self.__vector_variables





      def get_total_fitness(self):

          """

             Captura el Fitness total de la Población.

 

             :returns: El valor del Fitness poblacional.

             :rtype: Float

          """

         

          #Se regresa el valor concerniente al Fitness total de la población.

          return self.__total_fitness





      def set_total_fitness(self,value):

          """

             Actualiza el Fitness total de la Población.

 

             :param value: El valor a actualizar.



             :type value: Float

          """

         

          #Se actualiza el valor de la variable correspondiente.

          self.__total_fitness = value





      def get_total_expected_value(self):

          """

             Regresa el valor esperado de la Población.

 

             :returns: El valor esperado.

             :rtype: Float

          """

         

          #Se regresa el valor esperado de la población.

          return self.__total_expected_value





      def get_objective_functions_extreme_values(self):

          """

             Regresa el listado de los valores máximo y mínimo de las 

             funciones objetivo para el cálculo de sigma share.

 

             :returns: El listado con los valores máximo y mínimo para las

                       funciones objetivo.

             :rtype: List

          """

         

          #Se regresa el listado de las deltas.

          return self.__objective_functions_extreme_values





      def set_objective_functions_extreme_values(self,objective_functions_extreme_values):

          """

             Actualiza el listado de valores máximo y mínimo de las

             funciones objetivo para el cálculo de sigma share.

             

             :param objective_functions_extreme_values: Una lista con los valores máximo y mínimo

                                                        de cada una de las funciones objetivo.



             :type objective_functions_extreme_values: List

          """

         

          #Se actualiza el valor de la variable en cuestión.

          self.__objective_functions_extreme_values = objective_functions_extreme_values

 



      def get_decision_variables_extreme_values(self):

          """

             Regresa el listado de los valores máximo y mínimo de las 

             variables de decisión para el cálculo de sigma share.

 

             :returns: Una colección con los valores máximo y mínimo para las

                       variables de decisión.

             :rtype: Dictionary

          """

         

          #Se regresa el listado de las deltas.

          return self.__decision_variables_extreme_values





      def set_decision_variables_extreme_values(self,decision_variables_extreme_values):

          """

             Actualiza el listado de valores máximo y mínimo de las

             variables de decisión para el cálculo de sigma share.

             

             :param decision_variables_extreme_values: Un conjunto con los valores máximo y mínimo

                                                       de cada una de las variables de decisión.



             :type decision_variables_extreme_values: Dictionary

          """

         

          #Se actualiza el valor de la variable en cuestión.

          self.__decision_variables_extreme_values = decision_variables_extreme_values

 



      def add_individual(self,position,number_of_columns,complete_chromosome,complete_chromosome_length):

          """

             Añade un individuo a la Población.

              

             :param position: La posición dentro del arreglo de individuos 

                              donde se colocará el nuevo elemento.

             :param complete_chromosome: El cromosoma del Individuo.



             :type position: Integer

             :type complete_chromosome: Array

          """



          #Se agrega dentro del arreglo de individuos la instancia del individuo nuevo.

          #Dado que todos los atributos para crear una instancia ya han sido capturados

          #sólo se necesita el nuevo cromosoma. Como se puede apreciar, este método en 

          #realidad sólo sustituye elementos, no agrega posiciones extra para colocas

          #nuevos individuos.

		  

          self.__population[position] = Individual(

                                                   complete_chromosome,

                                                   complete_chromosome_length,

                                                   self.__predictive_variable,

                                                   self.__training_set,

                                                   self.__testing_set,

                                                   number_of_columns,

                                                   self.__available_global_columns,

                                                   self.__forbidden_columns,       

                                                  )    



                    

      def calculate_population_properties(self):

          """

             Calcula atributos individuales con base en los valores de toda la Población.

          """



          #Por cada individuo se hace lo siguiente:

          for individual in self.__population:



              #Se calcula el valor esperado de la población, el cual consiste en: 

              #fitness/(fitness total/tamaño de la población)

              try:

                  expected_value = individual.get_fitness()/(self.__total_fitness/self.__population_size)



              except:

                  expected_value = 0.0



              #Se agrega este valor calculado en el individuo actual.

              individual.set_expected_value(expected_value)



              #Se calcula el valor esperado de la población

              self.__total_expected_value += expected_value     





      def sort_individuals(self,method,is_descendent):

          """

             Ordena los Individuos de acuerdo a algún criterio dado.

         

             :param method: El método o atributo sobre el cual se hará la comparación.

             :param is_descendent: Indica si el ordenamiento es ascendente o descendente.

 

             :type method: String

             :type is_descendent: Boolean

          """



          #Se ordena la población con base en "method" y el orden lo indica el atributo "is_descendant".

          self.__population.sort(key = operador.methodcaller(method),reverse = is_descendent)





      def shuffle_individuals(self):

          """

             Desordena los elementos de la Población.

          """



          #Se desordena la lista de individuos.

          aleatorio.shuffle(self.__population)





      def print_info(self):

          """

             Imprime en texto las características de los Individuos

             de la Población, tanto grupales como individuales **(en consola)**.

          """



          print("Total fitness: " + str(self.__total_fitness))

          print("Total expected value: " + str(self.__total_expected_value))

          print("Individuals: ")

          for x in range (self.__population_size):

              print("Number: " + str(x))



              #Se imprime cada individuo.

              self.__population[x].print_info()      





#*************************************************************************************************************************************



    

class Community:

      """

         | Proporciona toda la infraestructura lógica para poder construir poblaciones y operar con éstas,

          además de transacciones relacionadas con sus elementos de manera individual.

         | Se le llama Community porque aludiendo a su significado una Community **(ó Comunidad)**

          consta de al menos una Population **(o Población)**. De esta manera se deduce que en algún momento

          habrán métodos que involucren a más de una población.



         :param vector_variables: Lista que contiene las variables de decisión previamente 

                                  saneadas por Controller/Controller.py.

         :param sharing_function_parameters: Diccionario que contiene todos los parámetros adicionales a la técnica

                                             de Fitness seleccionada por el usuario.

         :param selection_parameters: Diccionario que contiene todos los parámetros adicionales a la técnica

                                      de selección **(Selection)** usada por el usuario.

         :param crossover_parameters: Diccionario que contiene todos los parámetros adicionales a la técnica

                                      de cruza **(Crossover)** manejada por el usuario.

         :param mutation_parameters: Diccionario que contiene todos los parámetros adicionales a la técnica

                                     de mutación **(Mutation)** seleccionada por el usuario.

         

         :type vector_variables: List

         :type representation_parameters: Dictionary

         :type sharing_function_parameters: Dictionary

         :type selection_parameters: Dictionary

         :type crossover_parameters: Dictionary

         :type mutation_parameters: Dictionary

         :returns: Community

         :rtype: Instance

      """



      def __init__(

                   self,

                   vector_variables,

                   sharing_function_parameters,

                   selection_parameters,

                   crossover_parameters,

                   mutation_parameters,

                   predictive_variable,

                   training_set,

                   testing_set,

                   available_global_columns,

                   forbidden_columns

                   ):



          #Se almacenan todos los elementos en atributos privados para

          #su posterior uso. Dado que se crea una Community por cada MOEA, conviene

          #tener las características irrepetibles para un desempeño mayor.

          self.__vector_variables = vector_variables

          self.__sharing_function_parameters = sharing_function_parameters

          self.__selection_parameters = selection_parameters

          self.__crossover_parameters = crossover_parameters

          self.__mutation_parameters = mutation_parameters

          

          self.__predictive_variable = predictive_variable

          self.__training_set = training_set

          self.__testing_set = testing_set

          self.__available_global_columns = available_global_columns

          self.__forbidden_columns = forbidden_columns

            

          self.__chromosome_size = len(available_global_columns)            

          

          #Elemento que albergará el tamaño de cromosomas por cada función objetivo.

          self.__length_chromosomes = []



          #Se añade el vector de variables a los parámetros de Sharing Function, ya que este valor se utilizará

          #para el cálculo de Sigma Share (véase Model/SharingFunction).         

          self.__sharing_function_parameters["vector_variables"] = self.__vector_variables



      

      def __create_chromosome(self):

          """

          """

          number_of_columns = 0

          chromosome = ""

    

          for x in range (self.__chromosome_size):

             current_gene = str(aleatorio.randint(0,1))

             chromosome += current_gene

        

             if current_gene == "1":

                number_of_columns += 1

             

          return number_of_columns, chromosome





      def get_chromosome_size(self):

          """

          """

          return self.__chromosome_size





      def init_population(self,population_size):

          """

             Crea una población de manera aleatoria.



             :param population_size: El tamaño de la población. 



             :type population_size: Integer

             :returns: Population

             :rtype: Instance

          """



          #Se ejecuta la función "calculate_length_subchromosomes", la cual regresa el tamaño del cromosoma

          #por cada función objetivo, creando así un super cromosoma que constará de todos los tamaños de los subcromosomas. 

          #El resultado depende de la técnica de representación utilizada (véase Model/ChromosomalRepresentation). 

          self.__length_subchromosomes = [self.__chromosome_size]



          #Se agrega el tamaño de los subcromosomas como parámetro adicional para las técnicas de Sharing Function,

          #de la cual se hablará más adelante.

          self.__sharing_function_parameters["length_subchromosomes"] = self.__length_subchromosomes



          #A continuación se crea una instancia de la clase Population, cuyos elementos aún no se inicializan.

          population = Population(

                                  population_size,

                                  self.__vector_variables,

								  self.__predictive_variable,

								  self.__training_set,

								  self.__testing_set,

								  self.__available_global_columns,

								  self.__forbidden_columns

                                 )



          #Entonces, con base en el resultado de self.__length_subchromosomes se inicializan los Individuos

          #de la Población, indicando además el tamaño de cada subcromosoma y por ende, el tamaño del super

          #cromosoma.

          for x in range (population_size):



              #Se manda llamar la función "create_chromosome" dependiendo de la representación elegida (véase

              #Model/ChromosomalRepresentation).

              number_of_columns,complete_chromosome = self.__create_chromosome()

              

              #Se crea un individio con base en el cromosoma creado.

              population.add_individual(x,number_of_columns,complete_chromosome,self.__chromosome_size)

  

          return population





      def create_population(self,set_chromosomes):

          """

             Crea una población usando un conjunto de cromosomas como base.



             :param set_chromosomes: Conjunto de cromosomas. 



             :type set_chromosomes: List

             :returns: Population

             :rtype: Instance

          """



          #Se crea una instancia de la población (Population) con sus elementos aún no inicializados.

          population = Population(

                                  len(set_chromosomes),

								  self.__vector_variables,

		                          self.__predictive_variable,

								  self.__training_set,

								  self.__testing_set,

								  self.__available_global_columns,

								  self.__forbidden_columns

                                  )



          #Se inicializan cada individuo con un elemento del conjunto de cromosomas.

          for x in range(len(set_chromosomes)):

              number_of_columns = 0

              current_chromosome = set_chromosomes[x]

              

              for gene in current_chromosome:

                  if gene == "1":

                     number_of_columns += 1 

                     

              population.add_individual(x,number_of_columns,current_chromosome,self.__chromosome_size)



          return population



    

      def evaluate_population_functions(self,population):

          """

             | Evalúa cada uno de los subcromosomas de los individuos de la 

              población **(Population)**.

             | De manera adicional obtiene el listado de los valores extremos tanto

              de variables de decisión como de funciones objetivo para el 

              cálculo del sigma share **(véase el método __using_sharing_function)**. 

           

             :param population: La población sobre la que se hará la operación. 

             :type population: Instance

          """



          #Se obtienen las estructuras donde se almacenarán los valores mínimo y máximo para cada variable

          #de decisión y lo mismo para cada función objetivo respectivamente (estos valores se suelen utilizar

          #para el cálculo del sigma share, el cual es un factor determinante en el Sharing Function).

          decision_variables_extreme_values = population.get_decision_variables_extreme_values()

          objective_functions_extreme_values = population.get_objective_functions_extreme_values()

          

          #Se toman los individuos de la Población.

          individuals = population.get_individuals()

          for individual in individuals:



              #Por cada Individuo se toma su correspondiente cromosoma (que será en realidad el súper cromosoma).

              #complete_chromosome = individual.get_complete_chromosome()          



              #A continuación se devuelven las variables de decisión evaluadas por cada individuo con ayuda del método

              #"evaluate_subchromosomes". La obtención de las variables de decisión dependerá de la 

              #representación usada (véase Model/ChromosomalRepresentation).

              #decision_variables = getattr(self.__representation_instance,"evaluate_subchromosomes")(complete_chromosome,self.__length_subchromosomes,self.__vector_variables,self.__number_of_decimals,self.__representation_parameters)



              #Al final en el Individuo se evalúan las funciones objetivo con base en las variables de decisión recién obtenidas.

              #(véase la clase Individual).

              individual.evaluate_functions()



              #Por cada Individuo se obtienen sus variables de decisión evaluadas, así como las

              #respectivas funciones objetivo.

              current_evaluated_variables = individual.get_decision_variables()

              current_evaluated_functions = individual.get_evaluated_functions()



              for x in range (1):

                  #----------------------------------------------------------------------------------



                  #Se obtiene el valor de la función objetivo actual.

                  current_decision_variable_value = current_evaluated_variables[x] 



                  #A continuación se almacenan los valores mínimo y máximo actuales

                  #de la variable de decisión actual.

                  current_extreme_values = decision_variables_extreme_values[x]

                  current_minimal_value = current_extreme_values[0]

                  current_maximal_value = current_extreme_values[1]



                  #Si el valor actual es menor al que estaba guardado, se hace la actualización

                  #correspondiente para el valor mínimo.

                  if current_decision_variable_value < current_minimal_value:

                     decision_variables_extreme_values[x][0] = current_decision_variable_value 



                  #Si el valor actual es mayor al que estaba guardado, se hace la actualización

                  #correspondiente para el valor máximo.

                  if current_decision_variable_value > current_maximal_value:

                     decision_variables_extreme_values[x][1] = current_decision_variable_value





              #Ahora se realiza la actualización para los valores mínimo y máximo de cada

              #función objetivo.

              for x in range (3):

               

                  #Se obtiene el valor de la función objetivo actual.

                  current_function_value = current_evaluated_functions[x] 



                  #A continuación se almacenan los valores mínimo y máximo actuales

                  #de la variable de decisión actual.

                  current_extreme_values = objective_functions_extreme_values[x]

                  current_minimal_value = current_extreme_values[0]

                  current_maximal_value = current_extreme_values[1]



                  #Si el valor actual es menor al que estaba guardado, se hace la actualización

                  #correspondiente para el valor mínimo.

                  if current_function_value < current_minimal_value:

                     objective_functions_extreme_values[x][0] = current_function_value 



                  #Si el valor actual es mayor al que estaba guardado, se hace la actualización

                  #correspondiente para el valor máximo.

                  if current_function_value > current_maximal_value:

                     objective_functions_extreme_values[x][1] = current_function_value

                     



          #Al final se reinsertan en la población los valores actualizados para los valores mínimos y máximos

          #de tanto las variables de decisión y las funciones objetivo.

          population.set_decision_variables_extreme_values(decision_variables_extreme_values)

          population.set_objective_functions_extreme_values(objective_functions_extreme_values)





      def __compare_dominance(self,current,challenger,allowed_functions):

          """

             .. note:: Este método es privado.

 

             Permite realizar la comparación de las funciones objetivo de los 

             individuos current y challenger tomadas una a una para indicar así quién es el dominado y quién

             es el que domina. Cabe mencionar que más apropiadamente se le conoce como dominancia

             fuerte de Pareto.



             :param current: El Individuo inicial para comprobar dominancia.

             :param challenger: El Individuo que reta al inicial para comprobar dominancia.

             :param allowed_functions: Lista que indica cuáles son las funciones objetivo que deben 

                                       compararse.



             :type current: Instance

             :type challenger: Instance

             :type allowed_functions: List

             :returns: True si current domina a challenger, False en otro caso.

             :rtype: Boolean

          """

          

          #Aquí se almacenará el resultado.

          result = False



          #Aquí se almacenan los contadores para cuando un valor es menor a otro (lt, less than)

          #y un valor es menor o igual a otro (let, less equal than).

          lt = 0

          let = 0



          #Se toman las funciones de ambos individuos.

          current_evaluated_functions = current.get_evaluated_functions()

          challenger_evaluated_functions = challenger.get_evaluated_functions()



          #Aquí se indica las posiciones de las funciones objetivo que se deben comparar en caso

          #de que allowed_functions NO contenga la palabra "All".

          if allowed_functions != "All":

             current_evaluated_functions = [current_evaluated_functions[i] for i in allowed_functions] 

             challenger_evaluated_functions = [challenger_evaluated_functions[i] for i in allowed_functions] 



          #A continuación se comṕaran las funciones objetivo tomadas una a una considerando ya las que

          #fueron filtradas por la lista allowed_functions.

          for x in range(len(current_evaluated_functions)):



                 #Se toman los valores correspondientes a la posición indicada.

                 current_value = current_evaluated_functions[x]

                 challenger_value = challenger_evaluated_functions[x]

                 

                 #Aquí radica la comparación fuerte de Pareto, para que current domine a challenger

                 #éste tiene que ser en todas sus funciones objetivo menor e igual con respecto a challenger

                 #y existir al menos una función objetivo en la que sea estrictamente menor.

                 

                 #Se busca la condición contraria, que current_value sea mayor que challenger,

                 #en este caso automáticamente se regresa un resultado Falso.

                 if current_value > challenger_value:

                    return result



                 #Si current_value y challenger_value son

                 #estrictamente menores se cumple la condición <   

                 if current_value < challenger_value:

                    lt += 1



                 #Si current_value y challenger_value son iguales 

                 #se cumple la condición de <=

                 if current_value <= challenger_value:

                    let += 1

                       

          #Para que se pueda considerar dominancia y dado que ya se consideró

          #el caso en que el current_value es mayor que challenger, basta con

          #verificar que los contadores correspondientes sean mayores que 0. 

          if lt > 0 and let > 0:

             result = True

    

          #Se regresa el resultado.

          return result

          

               

      def calculate_population_pareto_dominance(self,population,allowed_functions):

          """

             Realiza la comparación de dominancia entre todos los elementos de la Población con base

             en la evaluación de sus funciones objetivo.

             

             :param population: La Población sobre la que se hará la operación.

             :param allowed_functions: Lista que indica las funciones objetivo permitidas para hacer la 

                                       comparación.



             :type population: Instance

             :type allowed_functions: List

          """



          #Se toman los individuos de la Población.

          individuals = population.get_individuals()



          #A continuación se hace una comparación de todos los Individuos contra todos; es por ello que

          #se crea un ciclo anidado paara poder hacer tarl operación.

          for x in range(population.get_size()):



              #Se obtiene el individuo current

              current = individuals[x]

              

              for y in range(population.get_size()):     



                  #Aquí se garantiza que un mismo individuo no se puede comparar consigo mismo.

                  if y != x:



                     #Se obtiene el individuo challenger.

                     challenger = individuals[y]



                     #Se ejecuta la operación de comparación entre current y challenger.

                     dominance_condition = self.__compare_dominance(current,challenger,allowed_functions)

                    

                     #Si se llega al resultado True, significa que current domina a challenger

                     #o equivalentemente challenger es dominado por current, por lo cual se actualizan sus

                     #respectivos contadores que controlan el número de individuos que dominan y son dominados.

                     #(véase Model/Community/Population/Individual.py).

                     if dominance_condition == True:

                        #Se actualiza el valor de current que indica que ahora domina a uno más.

                        current.set_pareto_dominates(current.get_pareto_dominates() + 1)



                        #Se actualiza el valor de challenger que indica que ahora es dominado por uno más.

                        challenger.set_pareto_dominated(challenger.get_pareto_dominated() + 1)

 



      def assign_goldberg_pareto_rank(self,population,additional_info = False,allowed_functions = "All"):

          """

             | Asigna una puntuación **(ó rank)** a cada uno de los Individuos de una Población con base en su dominancia

              de Pareto.

             | En términos generales, el algoritmo trabaja con niveles, es decir, primero toma los Individuos no

              dominados y les asigna un valor 0, luego los elimina del conjunto y nuevamente aplica la 

              operación sobre los no dominados del nuevo conjunto, a los que les asigna el valor 1, y así

              sucesivamente hasta no quedar Individuos.

             | Esta técnica es usada principalmente por N.S.G.A. II.



             :param population: La Población sobre la que se hará la operación.

             :param additional_info: Un valor que le indica a la función que debe regresar información 

                                     adicional.

             :param allowed_functions: Lista que contiene las posiciones de las funciones que son admisibles 

                                       para hacer comparaciones. Por defecto tiene el valor "All".

 

             :type population: Instance

             :type additional_info: Boolean

             :type allowed_functions: List

             :returns: Si additional_info es True: un arreglo con dos elementos: en el primero 

                       se almacena una lista con los niveles de dominancia disponibles, mientras que el 

                       segundo consta de una estructura que contiene todos los posibles niveles y asociados 

                       a éstos, los cromosomas de los Individuos que los conforman.

                       Si additional_info es False: el método es void **(no regresa nada)**.

             :rtype: List

          """



          #Contiene los cromosomas de los Individuos del nivel 0.

          f1 = []

          

          #Contiene los identificadores asociados a los Individuos que conforman el 

          #nivel "i" actual.

          current_fi = []



          #Aquí se alojarán los niveles de dominancia que no estén vacíos.

          pareto_fronts_list = []



          #En esta estructura se almacenarán los identificadores de los Individuos que están

          #siendo dominados por un Individuo "p".

          sp = {}

          

          #Se crea una estructura donde se guardarán los niveles, donde para cada nivel se almacenan

          #los cromosomas de los Individuos que constituyen cada nivel.

          pareto_fronts = {}

      

          #Se obtiene el tamaño de la Población.

          population_size = population.get_size()

 

          #Se obtienen los Individuos de una Población.

          individuals = population.get_individuals()



          #Esta variable almacena el nivel de dominancia actual.

          current_front = population_size



          #Por cada Individuo en la Población se hace lo siguiente.

          for x in range (population_size):



              #Este número indicará el número de soluciones que 

              #dominan a una solución "x".

              np = 0



              #Se crea en la estructura apropiada una referencia de 

              #los Individuos que dominará la solución "x".

              sp[x] = []



              #Se obtiene el Individuo actual.

              current_individual = individuals[x]



              #Se verifica el proceso de dominancia con los demás Individuos.

              for y in range (population_size):



                  #La dominancia no tiene sentido para el mismo Individuo, de modo que

                  #se descarta.

                  if x != y:

                     

                     #Se obtiene el Individuo que será comparado con el Individuo x.

                     challenger = individuals[y]



                     #Se verifica la dominancia de x con y. De ser positiva la operación, se agrega

                     #el identificador "y" a la lista de los elementos dominados por el Individuo "x".

                     if self.__compare_dominance(current_individual,challenger,allowed_functions) == True:

                        sp[x].append(y)



                     #En caso de ser negativo se verifica que "y" domine a "x", por lo que de ser 

                     #verdadero se incrementa el número de soluciones que dominan a "x".

                     elif self.__compare_dominance(challenger,current_individual,allowed_functions) == True:        

                          np += 1



              #Se actualiza la información del número de dominados para una solución "x"

              #en la estructura apropiada.

              individuals[x].set_pareto_dominated(np)



              #Se busca el valor np más bajo, ya que no necesariamente es 0.

              if np < current_front:

                 current_front = np 

              

          #Se obtienen los Invididuos cuyo np haya sido el más bajo,

          #esto corresponde al primer nivel de dominancia 0.

          for identifier in range(population_size):

             

              #Se obtiene el Individuo actual.

              current_individual = individuals[identifier]



              #Si el número de individuos que dominan al Individuo actual

              #es el mínimo entonces se agrega en la estructura f1.

              if current_individual.get_pareto_dominated() == current_front:



                 #Se actualiza el ranking del Individuo de la estructura f1 al 

                 #frente actual + 1                   

                 current_individual.set_rank(current_front + 1)



                 #Se añade el cromosoma correspondiente a la estructura f1.

                 f1.append(current_individual.get_complete_chromosome())      



                 #Se añade el identificador del Individuo correspondiente a la 

                 #estructura f1.

                 current_fi.append(identifier)

                 

          #El frente de dominancia de nivel 0 es el que se conforma con los cromosomas de los 

          #Individuos que no están dominados.

          pareto_fronts[current_front] = f1



          #Se añade a la lista el nivel de dominancia inicial.

          pareto_fronts_list.append(current_front)

          

          #A continuación se incrementa el nivel de dominancia actual en una unidad.

          current_front += 1

           

          #Mientras la lista con los identificadores no sea vacía

          #se hace lo siguiente:

          while current_fi != []:

                #print("entro en el fi porque hay cosas"

                #Las siguientes estructuras albergarán

                #los identificadores y cromosomas de los Individuos

                #de los siguientes niveles de dominancia respectivamente.

                h_ids = []

                h_chromosomes = []



                #Se toman los identificadores del conjunto actual. 

                for z in current_fi:



                    #Se obtiene el conjunto de identificadores relativos a las 

                    #soluciones que domina el Individuo asociado al identificador "z".

                    current_sp = sp[z]



                    #Se usa cada identificador "k".

                    for k in current_sp:



                        #A continuación se obtiene el Individuo correspondiente

                        #al identificador "k".

                        q = individuals[k]#population_dict[k]



                        #Se disminuye en una unidad el valor de los Individuos

                        #que dominan a q.

                        nq = q.get_pareto_dominated() - 1 

                        q.set_pareto_dominated(nq)



                        #Si dicho valor es 0 significa que es parte del nivel de dominancia actual

                        #por lo que debe de agregarse su cromosoma en la estructura h_chromosome y 

                        #su identificador en h_id.

                        if nq == 0:

                      

                           #Se actualiza el número de elementos que dominan a la solución actual.

                           q.set_pareto_dominated(current_front)



                           #Se actualiza el rango actual

                           q.set_rank(current_front + 1)



                           #Se agrega el identificador.

                           h_ids.append(k)

 

                           #Se agrega el cromosoma.

                           h_chromosomes.append(q.get_complete_chromosome())



                #Puede darse el caso en que un nivel de dominancia se encuentre vacío, 

                #por ello es que se verifica que el nivel actual no esté vacío.                                  

                if h_chromosomes != []:



                   #Se agrega el nivel actual con el resultado de la lista h_chromosome

                   pareto_fronts[current_front] = h_chromosomes



                   #Este valor se agrega a una lista que indica que un nivel de dominancia no está

                   #vacío.

                   pareto_fronts_list.append(current_front)



                #Se actualiza el nivel de dominancia actual.

                current_front += 1 

                   

                #La estructura de identificadores pasa a ser el nuevo current_fi para que puedan

                #verificarse nuevos elementos.

                current_fi = h_ids

        

          #Al final se regresa la lista que contiene la lista de niveles de dominancia disponibles

          #y la estructura con los niveles de dominancia y sus respectivos elementos.

          if additional_info == True:

             return [pareto_fronts_list,pareto_fronts]





      def assign_population_fitness(self,population):

          """

             Se implementa la asignación de Proportional Fitness **(ó Fitness Proporcional)**

             con base en la información especificada con anterioridad.

          """



          #Este valor almacenará la suma de F0 de cada Individuo.

          total_values = 0.0



          #Este valor almacenará el Fitness total de la Población.

          total_fitness = 0.0



          #Se obtiene el número de Individuos de la Población.

          population_size = population.get_size()



          #Primero se obtiene la suma de F0 de todos los Individuos.

          for individual in population.get_individuals():

              total_values += population_size - individual.get_rank()



          #Usando la cantidad anterior, se hace lo siguiente para cada Individuo:

          for individual in population.get_individuals():



              #Se calcula el Fitness de manera proporcional.

              current_fitness = (population_size - individual.get_rank())/total_values



              #Se actualiza el Fitness del Individuo.

              individual.set_fitness(current_fitness)



              #Se agrega el Fitness actual al total de Fitness de la Población.

              total_fitness += current_fitness



 

          #Se actualiza el Fitness de la Población.   

          population.set_total_fitness(total_fitness)



          #Se actializan propiedades relativas al Fitness para cada Individuo.

          population.calculate_population_properties()

    

    

      def __calculate_distance(self,individual_i,individual_j,sharing_function_parameters):

          """

             Con base en la información proporcionada anteriormente, se implementa

             el cálculo de la distancia entre dos Individuos apoyándose de la técnica

             conocida como Distancia de Hamming **(ó Hamming Distance)**.

          """



          #Aqui se almacenará la distancia de Hamming.

          hamming_distance = 0.0

    

          #Se toman los cromosomas de los Individuos participantes.

          chromosome_i = individual_i.get_complete_chromosome()

          chromosome_j = individual_j.get_complete_chromosome()

    

          #Se obtiene la longitud de una de las cadenas cromosómicas, la cual es la misma

          #en ambos Individuos.

          length_chromosome = len(chromosome_i)



          #Por cada gen en los cromosomas se hace lo siguiente:

          for x in range (length_chromosome):

 

              #Se obtiene el gen de cada Individuo localizado en la misma posición

              #en el cromosoma.

              gen_i = chromosome_i[x]

              gen_j = chromosome_j[x]



              #Se realiza la comparación pertinente, si sus alelos son diferentes

              #se actualiza el valor de la Distancia de Hamming.

              if gen_i != gen_j:

                 hamming_distance += 1

      

          #Se regresa la distancia de Hamming.

          return hamming_distance

          

        

      def __calculate_sigma_share(population,sharing_function_parameters):

          """

             Basándose en las indicaciones mencionadas anteriormente, se

             lleva a cabo la implementación de la obtención del valor Sigma Share.

          """

        

          #Aquí se almacena el tamaño total del cromosoma.

          distance = 0.0



          #A continuación se obtienen las longitudes de los subcromosomas

          #(véase Model/ChromosomalRepresentation).

          length_subchromosomes = sharing_function_parameters["length_subchromosomes"]



          #Se obtiene de la sección View el parámetro asociado al porcentaje de tolerancia,

          #es decir, la tolerancia máxima de genes del cromosoma visto como porcentaje

          #para considerar a dos cromosomas dentro del mismo Niche.

          percentage_of_acceptance = sharing_function_parameters["percentage_of_acceptance"]

 

          #Se suman las longitudes de los subcromosomas y se almacena el resultado

          #en la variable distance.

          for current_length in length_subchromosomes: 

              distance += current_length



          #Se regresa el número máximo real de genes en el cromosoma que permiten

          #considerar una distancia como válida, es decir, que se encuentra en el mismo Niche.

          return int(distance*percentage_of_acceptance)





      def __using_sharing_function(self,individual_i,individual_j,alpha_share,sigma_share):

          """

             .. note:: Este método es privado.



             | Devuelve un valor que ayuda al cálculo del Sharing Function.

             | A grandes rasgos el sharing function sirve para hacer una selección más precisa de los

              mejores Individuos cuando se da el caso de que tienen el mismo número de Individuos dominados.



             :param individual_i: Individuo sobre el que se hará la operación.

             :param individual_j: Individuo sobre el que se hará la operación.

             :param alpha_share: El valor necesario para poder calcular la distancia entre Individuos.

             :param sigma_share: El valor necesario para poder calcular la distancia entre Individuos.

             

             :type individual_i: Instance

             :type individual_j: Instance

             :type alpha_share: Float

             :type sigma_share: Float

             :returns: El resultado que contribuirá al sharing function. 

             :rtype: Float

          """



          #Aquí se colocará el resultado del cálculo de la distancia

          result = 0.0



          #Se calcula la distancia entre los individuos usando la técnica que el usuario eligió en la sección gráfica

          #(véase Model/SharingFunction).

          dij = self.__calculate_distance(individual_i,individual_j,self.__sharing_function_parameters)



          #De acuerdo a la técnica, si la distancia resulta menor a sigma se hace la siguiente operación.

          if dij < sigma_share:

             result = 1.0 - (dij/sigma_share)**alpha_share



          #Se regresa el resultado

          return result

        



      def calculate_population_niche_count(self,population):

          """

             Calcula el valor conocido como niche count que no es mas que la suma de los sharing function

             de todos los individuos j con el individuo i, con i != j.



             :param population: Conjunto sobre el que se hará la operación.

             

             :type population: Instance

          """



          #De acuerdo al trabajo escrito y a la documentación el valor de alpha típicamente

          #se asigna a 1, no obstante en este proyecto se le da la libertad al usuario de

          #seleccionar el valor libremente. Aquí se obtiene dicho valor con base en la información

          #ingresada por el usuario.           

          alpha_share = self.__sharing_function_parameters["alpha_sharing_function"]



          #A continuación se hace el cálculo del Sigma Share para poder obtener el Niche Count,

          #esto tomando en cuenta el tipo de distancia que haya elegido el usuario.

          sigma_share = self.__calculate_sigma_share(self.__sharing_function_parameters)



          #Se aplica un recorrido de los Individuos de la Población con ellos mismos

          #para calcular el Niche Count de cada uno.

          for individual_i in population.get_individuals():

 

              #El valor mínimo de Niche Count para un Individuo será 1.

              result = 1.0

              for individual_j in population.get_individuals():



                  #Así se garantiza que no se hará sharing function de los individuos con ellos mismos.

                  if individual_i != individual_j:



                     #Se calcula el niche count para cada individuo_i de la población

                     result += self.__using_sharing_function(individual_i,individual_j,alpha_share,sigma_share)

        

              #Al final se añade este valor al individuo i

              individual_i.set_niche_count(result)





      def calculate_population_shared_fitness(self,population):

          """

             Calcula el Shared Fitness **(ó Fitness Compartido)** de cada uno

             de los Individuos de la Población.



             :param population: Conjunto sobre el que se hará la operación.

             

             :type population: Instance

          """

          

          for individual in population.get_individuals():

             

              #El cálculo del sharing function por individuo es: fitness / niche count. Se hace

              #esto por cada Individuo.

              individual.set_fitness(individual.get_fitness()/individual.get_niche_count())      

          

  

      def execute_selection(self,parents):

          """

             De acuerdo a la información provista anteriormente, se implementa

             el método conocido como Stochastic Universal Sampling **(ó Muestreo Estocástico Universal)**.

          """



          #Se crea una estructura para almacenar los cromosomas de los

          #Individuos seleccionados. A su vez se crea la variable Pointer.

          chromosome_set = []

          pointer = aleatorio.random()



          #Se inicializan las variables correspondientes a la acumulación de Pointers

          #y Valores Esperados respectivamente.

          cumulative_pointers = 0

          cumulative_expected_value = 0

    

          #Con este valores se seleccionarán los Individuos y determinará si son aptos para

          #la etapa de reproducción.

          population_count = 0

          population_selected = 0



          #Se obtiene el tamaño de la Población y los Individuos, no sin antes habiéndolos 

          #desordenado primero para garantizar una selección más justa.  

          population_size = parents.get_size() 

          parents.shuffle_individuals()

          individuals = parents.get_individuals()

    

          #Se toma el primer Individuo como referencia para poder comenzar la

          #operación.

          current_individual = individuals[0]      



          #El siguiente proceso se realizará hasta que se hayan seleccionado 

          #para la reproducción tantos Individuos como el tamaño de la Población.

          while population_selected < population_size:

                #Se toma el Valor Esperado del Individuo actual.

                current_expected_value = current_individual.get_expected_value()



                #Si el Pointer es mayor que el Valor Esperado del actual Individuo considerando los valores

                #acumulados entonces se actualiza el candidato, pues lo anterior indica que se debe tomar el

                #siguiente Individuo espaciado con el valor Pointer.

                if cumulative_pointers + pointer > cumulative_expected_value + current_expected_value:

                   #Se actualiza el apuntador al siguiente Individuo.

                   population_count += 1   



                   #Se actualiza el Valor Esperado Acumulado

                   cumulative_expected_value += current_expected_value



                   #Se selecciona el Individuo siguiente(se puede ver que usa el operador '%' 

                   #para hacer cíclica la elección en caso de que se haya agotado la lista previamente).

                   current_individual = individuals[population_count % population_size]

          

                #Independientemente de la operación anterior, se actualizan lod siguientes valores para permitir

                #seguir obteniendo Individuos hasta que se satisfaga la demanda.

                population_selected += 1

                cumulative_pointers += pointer



                #Se agrega el cromosoma del Individuo seleccionado actualmente.

                chromosome_set.append(current_individual.get_complete_chromosome())

             

          return chromosome_set





      def __execute_crossover(self,chromosome_a,chromosome_b,chromosome_parameters):

          """

             Usando como base la información proporcionada anteriormente, se implementa

             el método conocido como N-Points Crossover **(ó Cruza en 'N' Puntos)**.

          """

          

          #De la sección de parámetros se obtiene la probabilidad de cruza.

          crossover_probability = crossover_parameters["probability_crossover_general"]    

    

          #Se inicializan los cromosomas hijos, los cuales contendrán la información

          #de la cruza entre los padres.

          chromosome_child_1 = []

          chromosome_child_2 = []

           

          #Se crea el número aleatorio que servirá para verificar la probabilidad

          #de cruza.

          crossover_number = aleatorio.random()



          #Si el número creado anteriormente es menor o igual al parámetro de la

          #probabilidad de cruza, entonces se procede con la etapa de recombinación

          #genética.

          if crossover_number <= crossover_probability:

             #Al entrar en la etapa de recombinación genética, primero se averigua

             #el número de puntos de corte que se solicitarán para la operación.

             how_many_points = crossover_parameters["how_many_points_npoints_crossover"]



             #Se guardan referencias a los cromosomas originales para no modificarlos.

             my_chromosome_a = chromosome_a

             my_chromosome_b = chromosome_b



             #Aquí se almacenarán los bloques alternados de cada hijo.

             mixed_chromosome_1 = "" 

             mixed_chromosome_2 = ""



             #Variable que contiene el tamaño del cromosoma (para averiguar los puntos

             #de corte).

             length_chromosome = len(chromosome_a)



             #Aquí serán almacenados los puntos de corte.

             sections_list = []



             #Esta variable permite alternar bloques de una manera rápida.

             flag = 0

           

             #En caso de que no se cumpla la restricción de puntos de corte,

             #se lanza una excepción.

             if how_many_points > length_chromosome - 1:

                raise ValueError("Number of points ({0}) exceeds chromosome's length ({1})".format(how_many_points,length_chromosome)) 

      

             #Aquí se considera el caso en que el número de puntos de cruza sea

             #'n-1', se utiliza una lista por comprensión para llenar la lista

             #de puntos de corte más rápidamente"""

             if length_chromosome == how_many_points + 1:

                real_sections_list = [x for x in range(length_chromosome + 1)]



             #En caso de tratarse de una menor cantidad de puntos

             #se procede a seleccionar de manera aleatoria los puntos

             #de corte de acuerdo a la variable how_many_points.     

             else: 



                #Se agrega siempre el punto 0 para que tenga coherencia

                #la extracción de bloques.

                sections_list.append(0)

                sections_list.append(length_chromosome)

                how_many_points_auxiliar = how_many_points

        

                #El siguiente ciclo permite seleccionar los puntos de corte,

                #asegurándose de no repetirlos y/o proporcionar valores inválidos.  

                while how_many_points_auxiliar != 0:

                      number = 1 + aleatorio.randint(0,length_chromosome - 2)

                

                      #Aquí se verifica que los puntos de corte no estén repetidos.

                      if not(number in sections_list):

                         sections_list.append(number) 

                         how_many_points_auxiliar -= 1            

                   

                #Se ordenan los puntos de corte para poder extraer 

                #los bloques más fácilmente.

                real_sections_list = sorted(sections_list)

          

             #Una vez creada la lista de puntos de corte, se procede a crear a los hijos.

             #Para ello se toman porciones de acuerdo a los índices de la lista.

             for x in range(1,len(real_sections_list)):

              

                 #Esta sección permite la alternancia de los bloques definidos por los puntos

                 #de corte.

                 if flag == 0:

                    mixed_chromosome_1 += my_chromosome_a[real_sections_list[x-1]:real_sections_list[x]]

                    mixed_chromosome_2 += my_chromosome_b[real_sections_list[x-1]:real_sections_list[x]]

                                   

                 elif flag == 1:

                      mixed_chromosome_1 += my_chromosome_b[real_sections_list[x-1]:real_sections_list[x]]

                      mixed_chromosome_2 += my_chromosome_a[real_sections_list[x-1]:real_sections_list[x]]

                    

                 #Esta variable permite aplicar la alternancia de bloques tantas

                 #veces como sea necesario.

                 flag = (flag + 1) % 2        

 

             #Se actualizan las variables destinadas a los hijos una vez terminada la concatenación

             #de bloques.

             chromosome_child_1 = mixed_chromosome_1

             chromosome_child_2 = mixed_chromosome_2

      

          #Si el número creado para la probabilidad de cruza es mayor que el parámetro

          #de probabilidad de cruza entonces no se aplica ninguna operación y los hijos

          #resultan en copias idénticas de los padres.

          else:

              chromosome_child_1 = chromosome_a

              chromosome_child_2 = chromosome_b

              

          #Al final se regresa un arreglo conteniendo a los 2 hijos.

          return [chromosome_child_1,chromosome_child_2]

          

      

      def __execute_mutation(self,chromosome,mutation_parameters):

          """

             Usando la información mostrada anteriormente, se desarrolla la función

             conocida como Binary Mutation **(ó Mutación Binaria)**.

          """



          #Aquí se almacenará el cromosoma mutado.

          mutated_chromosome = ""  



          #Se obtiene el valor que representará a la probabilidad de Mutación

          #establecida por el usuario.

          mutation_probability = mutation_parameters["probability_mutation_general"]

    

          #Por cada gen en el cromosoma se realiza lo siguiente:

          for gen in chromosome: 



              #Se crea el número que servirá de verificación 

              #para la probabilidad de cruza.

              number = aleatorio.random()



              #Si el número en cuestión es menor o igual que el parámetro de la

              #probabilidad de cruza entonces se procede a cambiar el alelo asociado

              #al gen actual.

              if number <= mutation_probability:

           

                 #Dado que se trata de una representación Binaria, la transformación

                 #es muy simple, si hay un 0 entonces el nuevo alelo asociado al gen se

                 #transfomará en 1 y viceversa.

                 if gen == "0":

                    mutated_chromosome += "1"

        

                 elif gen == "1":

                    mutated_chromosome += "0"

    

              #En caso de que la operación inherente a la probabilidad de cruza no

              #se haya cumplido, el gen actual no se modifica.     

              else:

                  mutated_chromosome += gen       

             

          #Finalmente se regresa el cromosoma mutado. 

          return mutated_chromosome





      def execute_crossover_and_mutation(self,selected_parents_chromosomes):

          """

             Realiza la cruza y mutación de los individuos. Para el caso de la cruza ésta se lleva a cabo siempre

             entre dos individuos, mientras que la mutación es unaria.



             :param selected_parents_chromosomes: El conjunto de cromosomas sobre los cuales se aplicarán dichos operadores genéticos.

            

             :type selected_parents_chromosomes: List

             :returns: Una instancia del tipo Model.Community.Population.

             :rtype: Instance   

          """



          #Se toma el tamaño de la población (que es el equivalente a tomar el tamaño de los individuos seleccionados), también

          #se inicializa una población para que ahí se almacenen los hijos mutados.

          size = len(selected_parents_chromosomes)          

          children = Population(

		                        size,

		                        self.__vector_variables,

		                        self.__predictive_variable,

								self.__training_set,

								self.__testing_set,

								self.__available_global_columns,

								self.__forbidden_columns

							    )



          #Si se tiene una población impar simplemente se añade un elemento al azar de los seleccionados automáticamente

          #a la siguiente generación no sin antes haber sido mutado.

          if size % 2 != 0:

             size -= 1  

             index = aleatorio.randint(0,size)

             lucky_chromosome = selected_parents_chromosomes[index]

             selected_parents_chromosomes.remove(selected_parents_chromosomes[index])

             modified_lucky_chromosome = self.__execute_mutation(lucky_chromosome,self.__mutation_parameters)

             

             number_of_columns = 0

             for gene in modified_lucky_chromosome:

                 if gene == "1":

                    number_of_columns +=1

                    

             children.add_individual(size,number_of_columns,modified_lucky_chromosome,self.__chromosome_size)

          

          #Tomando siempre un conjunto de cromosomas par, la cruza se realiza de la siguiente manera:

          count = 0

          for x in range(1,size,2):



              #Se toman dos cromosomas consecutivos.

              chromosome_a = selected_parents_chromosomes[x - 1]

              chromosome_b = selected_parents_chromosomes[x]



              #Se realiza la cruza sobre éstos, usando la instancia que se creó previamente con la técnica de cruza seleccionada

              #(véase Model/Operator/Crossover y Controller/Verifier.py), así como los parámetros que

              #se guardaron en la definición de la clase; la técnica de cruza devolverá 2 hijos.

              [child_1,child_2] = self.__execute_crossover(chromosome_a,chromosome_b,self.__crossover_parameters)



              #Ahora cada hijo es mutado de manera individual, utilizando una instancia de la técnica de mutación que fue elegida

              #por el usuario en la sección gráfica (véase Model/Operator/Mutation y Controller/Verifier.py) y los parámetros

              #que fueron guardados al inicio de la declaración de la clase.

              modified_child_1 = self.__execute_mutation(child_1,self.__mutation_parameters)

              modified_child_2 = self.__execute_mutation(child_2,self.__mutation_parameters)



              number_of_columns_1 = 0

              number_of_columns_2 = 0

              

              for y in range (len(modified_child_1)):

                  if modified_child_1[y] == "1":

                     number_of_columns_1 +=1



                  if modified_child_2[y] == "1":

                     number_of_columns_2 +=1



              #Se agregan los cromosomas a la población creada con anterioridad.

              children.add_individual(x - 1,number_of_columns_1,modified_child_1,self.__chromosome_size)

              children.add_individual(x,number_of_columns_2,modified_child_2,self.__chromosome_size)

              count +=2



          return children

   

      

      def get_best_individual(self,population):

          """

             Obtiene el mejor individuo dentro de una población. Para estos fines el mejor individuo es aquél que

             tenga mejor dominancia.



             :param population: La población sobre la cual se hará la búsqueda.

            

             :type population: Instance

             :returns: El individuo que cumple con la característica de la mayor dominancia.

             :rtype: Instance    

          """



          #Se guarda una copia de la población para no alterar la original. 

          sorted_population = population



          #Se manda llamar a un método de la población que ordena los individuos de acuerdo a algún criterio

          #(véase Model/Community/Population.py). El parámetro False determina el orden descendente del ordenamiento.

          sorted_population.sort_individuals("get_pareto_dominated",False)



          #Se toma el primer individuo de los individuos.

          individuals = sorted_population.get_individuals()

          best_individual = individuals[0]

          return best_individual





      def __get_best_individual_results(self,population):

          """

             .. note:: Este método es privado.

             

             Obtiene los valores de las variables de decisión y de las funciones objetivo

             por cada individuo.



             :param population: Una lista que contiene los mejores individuos por generación.



             :type population: List

             :returns: Una lista que contiene por un lado la tupla (generacion, funciones)

                       y por otro la tupla (generación, variables). Esto por cada generación.

             :rtype: List   

          """



          #Se crean los elementos donde al final se llenará la información.

          generations = []        

          decision_variables = []

          objective_functions = []

          queries = [] 



          #Por cada individuo se hace lo siguiente:

          for x in range (len(population)):

              individual = population[x]



              #Se agrega la generación

              generations.append(x + 1)



              #Se agrega la función objetivo. 

              objective_functions.append(individual.get_evaluated_functions())



              #Se agrega la variable de decisión.

              decision_variables.append(individual.get_decision_variables())

        

              queries.append(individual.get_query())

			  

          #Se regresa la tupla (generaciones, funciones) y (generaciones, variables).

          return [generations,objective_functions],[generations,decision_variables],[generations,queries]





      def __get_pareto_results(self,population):

          """

             .. note:: Este método es privado.

             

             | Obtiene el frente de Pareto, el complemento del frente de Pareto y el óptimo de Pareto.

             | Para una mejor orientación léase la parte escrita del proyecto.



             :param population: La población sobre la cual se obtendrán estos elementos.



             :type population: Instance

             :returns: Una lista que contiene el frente de Pareto, su complemento y el óptimo de Pareto.

             :rtype: List   

          """

     

          #Se crean las estructuras donde se guardarán el frente de Pareto, el complemento del frente de Pareto

          #y el óptimo de Pareto.

          pareto_front = []

          pareto_optimal = []

          pareto_complement = []



          #Se toman los individuos de la población.

          #Además se toma un individuo de muestra.

          individuals = population.get_individuals()

          sample = individuals[0]

          

          #Con base en la muestra se crean casillas para cada una de las funciones objetivo para el

          #frente de Pareto y su complemento.

          for function in sample.get_evaluated_functions():

              pareto_front.append([])

              pareto_complement.append([]) 

                   

          #Con base en la muestra también se crean casillas para cada una de las variables de decisión

          #para el óptimo de Pareto.

          for variable in sample.get_decision_variables():

              pareto_optimal.append([])



          #Por cada individuo se hace lo siguiente:

          for individual in individuals:

              

              #Si el individuo no tiene elementos que lo dominen, significa que es parte

              #del frente de Pareto, por lo que entonces se hace lo siguiente:

              individual_functions = individual.get_evaluated_functions()

              if individual.get_pareto_dominated() == 0:  



                 #Primero se agregan sus evaluaciones en las funciones objetivo a la estructura

                 #del frente de Pareto.

                 for x in range(len(pareto_front)):

                     pareto_front[x].append(individual_functions[x])



                 #A continuación se agregan las evaluaciones en las variables de decisión 

                 #a la estructura del óptimo de Pareto.

                 individual_decision_variables = individual.get_decision_variables()

                 for x in range(len(pareto_optimal)):

                     pareto_optimal[x].append(individual_decision_variables[x])



              #Si el individuo tiene al menos algún elemento que lo domine, significa que es del complemento

              #del frente de Pareto, por lo que simplemente se agregan las funciones objetivo a la 

              #respectiva estructura.

              else:

                 for x in range(len(pareto_complement)):

                     pareto_complement[x].append(individual_functions[x])



          #Al final se regresa la tupla (frente de Pareto, complemento del frente de Pareto, óptimo de Pareto).

          return [pareto_front,pareto_complement,pareto_optimal]





      def get_results(self,best_individual_along_generations,external_set_population):

          """

             Recolecta la información y la almacena en una estructura que contiene dos categorías principales: 

             funciones objetivo y variables de decisión. Por cada una existen las subcategorías Pareto y mejor 

             individuo, en referencia al óptimo o frente de Pareto **(según corresponda)** y a los valores del mejor 

             individuo por generación **(véase View/Additional/ResultsGrapher/GraphFrame.py)**.



             :param best_individual_along_generations: Una lista que contiene los mejores individuos por generación.

             :param external_set_population: La población sobre la cual se efectuarán las operaciones.

 

             :type best_individual_along_generations: List

             :type external_set_population: Instance

             :returns: Un diccionario con los elementos mostrados en la descripción.

             :rtype: Dictionary  

          """



          #Se crea la estructura final donde se almacenará toda la información.

          information = {}          



          #Se obtienen los valores para los mejores individuos por generación.

          objective_functions, decision_variables, queries = self.__get_best_individual_results(best_individual_along_generations)



          #Se obtienen el frente de Pareto, su complemento y el óptimo de Pareto.

          #Por una petición de la Dra. Katya Rodríguez Vázquez se omite el complemento de Pareto en las

          #impresiones finales, por ello es que se solicitará el complemento aquí por si en algún momento el usuario lo

          #necesita pero no se va a utilizar en la sección de impresión (View/Additional/ResultsGrapher/GraphFrame.py)

          #dado que este método no regresará esa parte.

          pareto_front, pareto_complement, pareto_optimal = self.__get_pareto_results(external_set_population)



          #Se crea la primera categoría (funciones objetivo) de la información final y se llena con los datos mostrados a continuación.

          information["objective_functions"] = {           

                                                

                                                "best individual": {

                                                                    "functions": objective_functions,

                                                                   }

                                               }



          #Se crea la segunda categoría (variables de decisión) de la información final y se llena con los datos mostrados a continuación.

          information["decision_variables"] = {

                                               

                                               "best individual": {

                                                                   "variables": decision_variables,

                                                                  }

                                              }

											  

       	  #Se crea la segunda categoría (variables de decisión) de la información final y se llena con los datos mostrados a continuación.

          information["queries"] = {

                                              

                                               "best individual": {

                                                                   "queries": queries,

                                                                  }

                                              }



          return information





#*************************************************************************************************************************************



    

class VariableSelector:

    

    def __init__(self,

                 generations,

                 population_size,

                 vector_variables,

                 sharing_function_parameters,

                 selection_parameters,

                 crossover_parameters,

                 mutation_parameters,

                 predictive_variable,

                 test_size,

                 non_dummy_columns,

                 forbidden_columns,

                 ):

                 

        self.__comunidad = None

        self.__generations = generations

        self.__population_size = population_size

        self.__vector_variables = vector_variables

        self.__sharing_function_parameters = sharing_function_parameters

        self.__selection_parameters = selection_parameters

        self.__crossover_parameters = crossover_parameters

        self.__mutation_parameters = mutation_parameters

        self.__predictive_variable = predictive_variable

        self.__test_size = test_size

        self.__non_dummy_columns = non_dummy_columns

        self.__forbidden_columns = forbidden_columns

        self.__available_global_columns = None

        self.__training_set = None

        self.__testing_set = None

      

        self.__create_sets()



          

    def __create_sets(self):

        """

        """

        print("Creating sets....")  

        

        df_training=pd.read_pickle('/content/drive/My Drive/Special_Analysis/df_final_2.pickle') 

        

        df_training=df_training.drop(['PlayerKey_x', 'PlayerKey_y','GameID_x', 'GameID_y', 'PlayKey','DM_M1', 'DM_M7', 'DM_M28', 'DM_M42'], axis=1)

        

        #print(df_training.columns.values)

      

        df_training = df_training.fillna(0)

	   

        df_training_complete, df_testing = train_test_split(df_training,test_size=self.__test_size)



        df_training_complete=df_training_complete.fillna(0)

        df_testing=df_testing.fillna(0)

         

        self.__available_global_columns = df_training_complete.columns.values.tolist()



        self.__training_set = df_training_complete

        self.__testing_set = df_testing

        print("Sets created.")       

        

        

    def execute_moea(self):

        """

           En esta parte se lleva a cabo la implementación del M.O.E.A. denominado

           N.S.G.A. II **(Non-dominated Sorting Genetic Algorithm ó Algoritmo Genético 

           de Ordenamiento No Dominado)**.



           La forma de proceder del método es la siguiente:



           1.- Se crea una Población Padre **(de tamaño n)**, a la cual se le evalúan las funciones objetivo de sus Individuos, se les asigna un Ranking **(Goldberg)** y posteriormente se les otorga un Fitness.



           2.- Con base en la Población Padre se aplica el operador de Selección para elegir a los Individuos que serán aptos para reproducirse.



           3.- Usando a los elementos del punto 2, se crea una Población Hija **(de tamaño n)**. 

 

           4.- Se crea una súper Población **(llamémosle S, de tamaño 2n)** que albergará todos los Individuos tanto de la Población Padre como Hija; a *S* se le evalúan las funciones objetivo de sus Individuos, se les asigna un Ranking **(Goldberg)** y posteriormente se les otorga un Fitness. 



           5.- La súper Población *S* se divide en subcategorías de acuerdo a los niveles de dominancia que existan, es decir, existirá la categoría de dominancia 0, la cual almacena Individuos que tengan una dominancia de 0 Individuos **(ningún Individuo los domina)**, existirá la categoría de dominancia 1 con el significado análogo y así sucesivamente hasta haber cubierto todos los niveles de dominancia existentes.



           6.- Se construye la nueva Población Padre, pare ello constará de los Individuos de *S* donde la prioridad será el nivel de dominancia, es decir, primero se añaden los elementos del nivel 0,luego los del nivel 1 y así en lo sucesivo hasta haber adquirido n elementos.

               Se debe aclarar que la adquisición de Individuos por nivel debe ser total, esto significa que no se pueden dejar Individuos sueltos para el mismo nivel de dominancia. 



               Supongamos que a un nivel k existen tantos Individuos que su presunta adquisición supera el tamaño n, en este caso se debe hacer lo siguiente:

    

           6.1.- Se crea una Población provisional **(Prov)** con los Individuos del nivel k, se evalúan las funciones objetivo a cada uno de sus Individuos, se les asigna un Ranking **(Goldberg)** y posteriormente se les asigna el Fitness.



               Con los valores anteriores se calcula el Niche Count **(véase Model/SharingFunction)** de los Individuos; una vez hecho ésto se seleccionan desde Prov los Individuos faltantes con los mayores Niche Count, esto hasta completar el tamaño n de la nueva Población Padre.



           7.- Al haber conformado la nueva Población Padre, se evalúan las funciones objetivo de sus Individuos, se les asigna el Ranking correspondiente **(Goldberg)** y se les atribuye su Fitness.



           8.- Se repiten los pasos 2 a 7 hasta haber alcanzado el límite de generaciones **(iteraciones)**.



           | Como su nombre lo indica, la característica de este algoritmo es la clasificación 

             de los Individuos en niveles para su posterior selección.



           | Esto al principio propicia una Presión Selectiva moderada por la ausencia de elementos 

             con dominancia baja que suele existir en las primeras generaciones, sin embargo en iteraciones 

             posteriores se agudiza la Presión Selectiva ya que eventualmente la mayoría de los Individuos 

             serán alojados en las primeras categorías de dominancia, cubriendo casi instantáneamente 

             la demanda de Individuos necesaria en el paso 6, por lo que las categorías posteriores serán 

             cada vez menos necesarias con el paso de los ciclos.



           | Por otra parte la fusión de las Poblaciones en *S* garantiza que siempre se conserven a 

             los mejores Individuos independientemente de la generación transcurrida, a eso se le llama Elitismo.

           | Por cierto que en el algoritmo original no existe un nombre oficial para *S* sino más bien se señala como

             una estructura genérica, sin embargo se le ha formalizado con un identificador para guiar apropiadamente al 

             usuario en el flujo del algoritmo.

 

           | Para finalizar se señala que el uso del ranking de Goldberg **(véase Model/Community/Community.py)** 

             es indispensable.

        """



        print("Welcome to NSGA-II")     

        #Se crea una instancia de Community ya que la mayoría de los métodos auxiliares

        #residen allí.

        self.__comunidad = Community(self.__vector_variables,

                                     self.__sharing_function_parameters,

                                     self.__selection_parameters,

                                     self.__crossover_parameters,

                                     self.__mutation_parameters,

                                     self.__predictive_variable,

                                     self.__training_set,

                                     self.__testing_set,

                                     self.__available_global_columns,

                                     self.__forbidden_columns)



        #Se crea una estructura para almacenar al mejor Individuo por generación.

        best_individual_along_generations = []



        #Se crea la Población Padre.

        parents = self.__comunidad.init_population(population_size)



        print("Starting variable selection...")

        #try: 

            #Se evalúan las funciones objetivo de los Individuos de la Población

            #Padre.

        print("Step 1")

        self.__comunidad.evaluate_population_functions(parents)



        #Se asigna el Ranking (Goldberg) correspondiente a los Individuos de la

        #Población Padre. 

        print("Step 2")  

        self.__comunidad.assign_goldberg_pareto_rank(parents)



        #Usando el Ranking, se asigna el Fitness a los Individuos de la 

        #Población Padre.

        print("Step 3")

        self.__comunidad.assign_population_fitness(parents)

 

        #El siguiente procedimiento se realizará hasta haber alcanzado

        #el número límite de generaciones.

        for x in range (1,generations + 1):

                print("Generación :", x)



                #Se seleccionan los Individuos de la Población Padre elegidos para reproducirse-

                selected_parents_chromosomes = self.__comunidad.execute_selection(parents)

 

                #Con base en los seleccionados en el paso anterior, se crea la Población Hija.

                children = self.__comunidad.execute_crossover_and_mutation(selected_parents_chromosomes)



                #El primer paso del algoritmo consiste en fusionar las poblaciones Padre e

                #Hija en una súper Poblacion, para ello se hace lo siguiente:

                #Se crea una estructura para almacenar los cromosomas de los Individuos.

                parents_and_children = []



                #Los cromososmas de los Individuos de la Población Padre son añadidos

                #a dicha estructura.

                for parent in parents.get_individuals():

                    parents_and_children.append(parent.get_complete_chromosome())



                #Los cromososmas de los Individuos de la Población Hija son añadidos

                #a dicha estructura.

                for child in children.get_individuals():

                    parents_and_children.append(child.get_complete_chromosome())



                #La súper Población de tamaño 2n es creada. Como dato de implementación tiene sentido 

                #que la súper Poblacion tenga tamaño de 2n y entonces tenga una dominancia máxima de 2n - 1.

                new_population = self.__comunidad.create_population(parents_and_children)

 

                #Se evalúan las funciones objetivo de los Individuos de la súper

                #Población.

                self.__comunidad.evaluate_population_functions(new_population)



                #Se asigna el Ranking (Goldberg) a los Individuos de la 

                #súper Población.

                [auxiliar_pareto_fronts,pareto_fronts] = self.__comunidad.assign_goldberg_pareto_rank(new_population,True)



                #Dado que debe haber n seleccionados y la súper Población consta de 2n Individuos se debe aplicar

                #un filtro, de modo que esta estructura albergará a los elegidos.

                chosen = []



                #Ahora se van tomando los cromosomas que pertenezcan primero al nivel de dominancia 0 (no hay ningún 

                #Individuo que los domine), luego a los del nivel de dominancia 1 (hay 1 Individuo que los domina) y así 

                #sucesivamente hasta haber seleccionado n elementos.

                #Cabe mencionar que se deben seleccionar los elementos del nivel completo siempre y cuando su tamaño no

                #exceda n.

                #Con esta variable se obtiene el nivel actual de dominancia.

                current_front_index = 0



                while len(chosen) != parents.get_size():



                      #Se obtiene el de nivel de dominancia actual.

                      current_front = auxiliar_pareto_fronts[current_front_index]



                      #Se verifica que los Individuos del nivel actual no sobrepasen el tamaño n,

                      #en caso de no rebasar el límite se agregan todos a la estructura chosen.

                      if len (chosen) + len(pareto_fronts[current_front]) <= parents.get_size():

                         for current_chromosome in pareto_fronts[current_front]:

                             chosen.append(current_chromosome)

  

                      #En caso de que, al momento de seleccionar un nivel k de dominancia, los elementos de la Población

                      #excedan el tamaño n, entonces se hace lo siguiente:

                      else:



                           #Es menester mencionar que en algunos casos la diferencia puede ser mucha porque no se entró en 

                           #el primer if, lo que significa que de entrada los primeros niveles de dominancia pueden ser tener 

                           #demasiados Individuos, mas que n. Por esa razón con el transcurso de las generaciones, sólo

                           #se verificará el nivel 0 de dominancia, a lo más, el nivel 1 y todos esos Individuos

                           #se agregarán en esta sección de código.

                           #Aquí se calcula el número de individuos que exceden a la población.

                           difference = parents.get_size() - len(chosen)

                      

                           #Lo que se hará entonces es agregar todos los Individuos del nivel de dominancia actual

                           #hasta cumplir con el tamaño n.

                           #Para ello se debe crear una población Provisional aparte y

                           #asignarle un Niche Count para que puedan ser seleccionados los faltantes.  

                           provisional = self.__comunidad.create_population(pareto_fronts[current_front])

 

                           #Se evalúan las funciones objetivo de los Individuos de la Población

                           #provisional.

                           self.__comunidad.evaluate_population_functions(provisional)



                           #Se asigna el Ranking de Goldberg para los Individuos de la Población

                           #provisional.

                           self.__comunidad.assign_goldberg_pareto_rank(provisional)



                           #Con base en el Ranking se asigna un Fitness a dichos Individuos.

                           self.__comunidad.assign_population_fitness(provisional)



                           #Ahora se calcula el Niche Count de la Población.

                           self.__comunidad.calculate_population_niche_count(provisional)



                           #Con base en este valor se ordenan a los Individuos de manera

                           #ascendente (un menor niche count significa que una solución tiene 

                           #menos vecinos por ende es más probable que dicha solución

                           #sea no dominada).

                           provisional.sort_individuals("get_niche_count",False)



                           #Se toman los Individuos de la Población provisional.

                           individuals = provisional.get_individuals()



                           #A continuación se añaden los elementos faltentes a la estructura

                           #chosen.

                           for x in range (difference):

                               chosen.append(individuals[x].get_complete_chromosome())

                      current_front_index += 1



                #Se crea la nueva Población Padre asociada a los elementos 

                #de la estructura chosen.

                parents = self.__comunidad.create_population(chosen)



                #Se evalúan las funciones objetivo de los Individuos de la 

                #nueva Población Padre.

                self.__comunidad.evaluate_population_functions(parents)



                #Se asigna el Ranking (Goldberg) a los Individuos de la 

                #nueva Población Padre.

                self.__comunidad.assign_goldberg_pareto_rank(parents)



                #Se asigna el Fitness con base en el Ranking a los Individuos

                #de la nueva Población Padre.

                self.__comunidad.assign_population_fitness(parents)



                best = self.__comunidad.get_best_individual(parents)

                print(best.get_evaluated_functions())

                #print(best.get_query())

                #Se añade el mejor Individuo por generación a la estructura creada para tal fin.

                best_individual_along_generations.append(best)



        #except Exception as e:

                #En caso de un error interno las generaciones automáticamente llegan a su límite

                #para cerrar la ventana en la parte de View.

                #generations_queue.append((execution_task_count,generations)) 



                #Posteriormente se regresa el siguiente diccionario con la información relativa

                #al origen del error.

        #        error = {

        #                 "response": "ERROR",

        #                 "class": "NSGAII", 

        #                 "method": "execute_moea",

        #                 "message": "An error has occurred during execution of NSGAII algorithm",

        #                 "type": (str(e))

        #                }   



        #        return error



        #Los resultados tienen el formato precisado dentro de la función get_results que se encuentra

        #en la clase Community. Es sumamente importante que el usuario revise esta función

        #ya que de ésta depende la graficación de resultados (véase View/Additional/ResultsGrapher/ResultsGrapherToplevel.py).

        #El conjunto que almacena todos los Individuos para impresión de resultados es el de la Población Padre.

        results = self.__comunidad.get_results(best_individual_along_generations,parents)



        #Se regresan dichos resultados.

        return results



      

#---------------------------------------------



generations = 800

population_size = 100

#population_size = 4



#Como poner esto para el sigma share.

vector_variables = [],



sharing_function_parameters = {

                               "alpha_sharing_function":4,

                               "percentage_of_acceptance":0.4,

                              }

                              

selection_parameters = {}

crossover_parameters = {

                        "probability_crossover_general": 0.7,    

                        "how_many_points_npoints_crossover":1,

                       }

                       

mutation_parameters = {

                       "probability_mutation_general": 0.1,

                      }



predictive_variable = "Target"

test_size = 0.3



non_dummy_columns = [ ]



forbidden_columns = [

                     "Target"

                    ]



print("Welcome to variable selection")   



from google.colab import drive

drive.mount('/content/drive',force_remount=True)



variable_selector = VariableSelector(generations,

                                     population_size,

                                     vector_variables,

                                     sharing_function_parameters,

                                     selection_parameters,

                                     crossover_parameters,

                                     mutation_parameters,

                                     predictive_variable,

                                     test_size,

                                     non_dummy_columns,

                                     forbidden_columns,             

                                    )

                                     



results = variable_selector.execute_moea()



print(results["queries"]["best individual"]["queries"][-1])

#print(results["objective_functions"]["best individual"]["functions"][-1])



pkl_filename = "/content/drive/My Drive/Special_Analysis/results.pickle"

with open(pkl_filename, 'wb') as file:

    pickle.dump(results, file)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



#Linear Regression

cols_linear = [

               'x',

               'y',

               'RosterPosition_Linebacker',

               'RosterPosition_Running Back',

               'RosterPosition_Wide Receiver',

               'StadiumType_Indoors',

               'StadiumType_Outddors',

               'FieldType_Natural',

               'Weather_Clear',

               'Weather_Clear and warm',

               'Weather_Indoor',

               'Weather_Partly Cloudy',

               'Weather_Sun & clouds',

               'PlayType_Kickoff Returned',

               'PlayType_Punt Returned',

               'Position_CB',

               'Position_DT',

               'Position_LB',

               'Position_TE',

               'PositionGroup_DB',

               'event_pass_outcome_incomplete',

               'BodyPart_Ankle',

               'BodyPart_Knee'

              ]



#Logit

cols_logit = [

              's', 

              'SumResults', 

              'RosterPosition_Defensive Lineman', 

              'StadiumType_Dome', 

              'StadiumType_Open', 

              'StadiumType_Retractable Roof',  

              'Weather_Clear and warm', 

              'Weather_Clear skies', 

              'PlayType_Kickoff Returned', 

              'PlayType_Pass', 

              'PlayType_Rush', 

              'Position_CB', 

              'event_first_contact', 

              'event_line_set', 

              'event_pass_forward', 

              'event_penalty_declined', 

              'event_punt_land'

             ]



train, test = train_test_split(df_final_1, test_size=0.3)



X_train_rf=train[cols_logit]

Y_train_rf=train['Target']

X_test_rf=test[cols_logit]

Y_test_rf=test['Target']



logistic = LogisticRegression(solver = 'liblinear')

logistic.fit(X_train_rf, Y_train_rf)



X_train_rf=train[cols_linear]

Y_train_rf=train['Target']

X_test_rf=test[cols_linear]

Y_test_rf=test['Target']





random_forest = RandomForestClassifier(n_estimators=1)

random_forest.fit(X_train_rf, Y_train_rf)





pkl_filename = "/content/drive/My Drive/Special_Analysis/logistic_regression_object_model.pickle"

with open(pkl_filename, 'wb') as file:

    pickle.dump(logistic, file)



print("Logistic Score: ", logistic.score(X_test_rf, Y_test_rf))



pkl_filename = "/content/drive/My Drive/Special_Analysis/random_forest_object_model.pickle"

with open(pkl_filename, 'wb') as file:

    pickle.dump(random_forest, file)



print("Ranfom Forest: ", random_forest.score(X_test_rf, Y_test_rf))
feature_importance = abs(model.coef_[0])

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5



featfig = plt.figure()

featax = featfig.add_subplot(1, 1, 1)

featax.barh(pos, feature_importance[sorted_idx], align='center')

featax.set_yticks(pos)

featax.set_yticklabels(np.array(df_final.columns)[sorted_idx], fontsize=8)

featax.set_xlabel('Relative Feature Importance')



plt.tight_layout()   

plt.show()
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from scipy.stats import skew



%config InlineBackend.figure_format = 'png' #set 'png' here when working on notebook

%matplotlib inline
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import cross_val_score

from itertools import product



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, Y_train, scoring='neg_mean_squared_error', cv = 5))

    return(rmse)
alphas = [0.0005, 0.001, 0.01, 0.03, 0.05, 0.1]

l1_ratios = [1.5, 1.1, 1, 0.9, 0.8, 0.7, 0.5]
cv_elastic = [rmse_cv(ElasticNet(alpha = alpha, l1_ratio=l1_ratio)).mean() 

            for (alpha, l1_ratio) in product(alphas, l1_ratios)]
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

idx = list(product(alphas, l1_ratios))

p_cv_elastic = pd.Series(cv_elastic, index = idx)

p_cv_elastic.plot(title = "Validation - Just Do It")

plt.xlabel("alpha - l1_ratio")

plt.ylabel("rmse")
# Zoom in to the first 10 parameter pairs

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

idx = list(product(alphas, l1_ratios))[:10]

p_cv_elastic = pd.Series(cv_elastic[:10], index = idx)

p_cv_elastic.plot(title = "Validation - Just Do It")

plt.xlabel("alpha - l1_ratio")

plt.ylabel("rmse")
elastic = ElasticNet(alpha=0.0005, l1_ratio=0.5)
elastic.fit(X_train, Y_train)
coef = pd.Series(elastic.coef_, index = X_train.columns)
print("Elastic Net picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Elastic Net Model")
!pip install --target=$nb_path lime
def prob(data):

    return np.array(list(zip(1-random_forest.predict(data),random_forest.predict(data))))
import lime

import lime.lime_tabular

import numpy as np
explainer = lime.lime_tabular.LimeTabularExplainer(df_final[cols].astype(int).values,  

mode='classification',training_labels=df_final['Target'],feature_names=cols)
feat=cols
i = 1

exp = explainer.explain_instance(df_final.loc[i,feat].astype(int).values, prob, num_features=15)
exp.show_in_notebook(show_table=True)
from google.colab import drive

drive.mount('/content/drive',force_remount=True)
#@title



from datetime import datetime, timedelta

from google.colab import auth

auth.authenticate_user()



import gspread

from oauth2client.client import GoogleCredentials



gc = gspread.authorize(GoogleCredentials.get_application_default())
import pickle

import matplotlib

import numpy as np

import pandas as pd

import seaborn as sns

import statsmodels.api as sm

from functools import reduce

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn.metrics import pairwise_distances_argmin_min

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso, ElasticNetCV , ElasticNet



%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D

#sns.set_context('poster')

#sns.set_color_codes()

#plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

#plt.rcParams['figure.figsize'] = (16, 9)

#plt.style.use('ggplot')
df_injury = pd.read_csv('/content/drive/My Drive/NFL/InjuryRecord.csv')

df_player_track = pd.read_csv('/content/drive/My Drive/NFL/PlayerTrackData.csv')

df_play_list = pd.read_csv('/content/drive/My Drive/NFL/PlayList.csv')
df_player_track.shape
plt.rcParams['figure.figsize'] = [50, 10]
total = df_player_track.isnull().sum().sort_values(ascending=False)

porciento = (df_player_track.isnull().sum()/df_player_track.isnull().count()).sort_values(ascending=False)

dato_nulo = pd.concat([total, porciento], axis=1, keys=['Total', '%'])

dato_nulo.head(50)
print(df_player_track['o'].median(skipna=True))
print(df_player_track['dir'].median(skipna=True))
df_player_track['o'].fillna(179.91, inplace=True)

df_player_track['dir'].fillna(180.06, inplace=True)
total = df_player_track.isnull().sum().sort_values(ascending=False)

porciento = (df_player_track.isnull().sum()/df_player_track.isnull().count()).sort_values(ascending=False)

dato_nulo = pd.concat([total, porciento], axis=1, keys=['Total', '%'])

dato_nulo.head(50)
df_player_track['event'].value_counts().plot.bar(title='Freq dist of Play Details')

plt.xticks(rotation=45)

plt.tight_layout()

col_names = ['time','x', 'y', 'dis', 's', 'o','dir']



fig, ax = plt.subplots(len(col_names), figsize=(16,12))



for i, col_val in enumerate(col_names):



    sns.distplot(df_player_track[col_val], hist=True, ax=ax[i])

    ax[i].set_title('Freq dist '+col_val, fontsize=10)

    ax[i].set_xlabel(col_val, fontsize=8)

    ax[i].set_ylabel('Count', fontsize=8)



plt.show()

plt.tight_layout()
col_names = ['time','x', 'y', 'dis', 's', 'o','dir']



fig, ax = plt.subplots(len(col_names), figsize=(8,40))



for i, col_val in enumerate(col_names):



    sns.boxplot(y=df_player_track[col_val], ax=ax[i])

    ax[i].set_title('Box plot - {}'.format(col_val), fontsize=10)

    ax[i].set_xlabel(col_val, fontsize=8)



plt.show()
def percentile_based_outlier(data, threshold=95):

    diff = (100 - threshold) / 2

    minval, maxval = np.percentile(data, [diff, 100 - diff])

    return (data < minval) | (data > maxval)



col_names = ['time','x', 'y', 'dis', 's', 'o','dir']



fig, ax = plt.subplots(len(col_names), figsize=(8,40))



for i, col_val in enumerate(col_names):

    x = df_player_track[col_val][:1000]

    sns.distplot(x, ax=ax[i], rug=True, hist=False)

    outliers = x[percentile_based_outlier(x)]

    ax[i].plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)



    ax[i].set_title('Outlier detection - {}'.format(col_val), fontsize=10)

    ax[i].set_xlabel(col_val, fontsize=8)



plt.show()
f, ax = plt.subplots(figsize=(10, 8))

corr = df_player_track.corr()

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            cmap="Blues")
%matplotlib inline

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from sklearn.preprocessing import LabelEncoder

import re

from wordcloud import WordCloud, ImageColorGenerator

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
df_playlist_injury=pd.merge(df_injury,df_play_list,on=['PlayerKey','GameID','PlayKey'],how="inner",indicator=True)
df_playlist_injury.shape
df_playlist_injury.describe()
df_playlist_injury['Surface'].value_counts().plot.bar(title='Freq dist of Field_Surface')

plt.xticks(rotation=45)

plt.tight_layout()

print(df_playlist_injury['Surface'].value_counts())
table = pd.pivot_table(df_playlist_injury, values='PlayerKey', index=['Surface','BodyPart'], aggfunc=lambda x: x.count() )

table_2 = pd.pivot_table(df_playlist_injury, values='PlayerKey', index=['Surface'],columns='BodyPart', aggfunc=lambda x: x.count() )

table.unstack().plot(kind='bar', stacked=True,colormap="Blues",edgecolor ='black')

table.unstack()

plt.title('Players Injury Per Type of Field')

plt.xlabel('Type of Field')

plt.ylabel('No. Players')

plt.legend( ('Ankle', 'Foot','Knee'))

plt.grid()

print(table_2)
g = sns.catplot("BodyPart",

                col="Surface",

                data=df_playlist_injury, kind="count",

                hue='RosterPosition',

                height=12, aspect=.7);



g.fig.suptitle('Number of Player per Roster Position and grouped by injury in each type of field.')

plt.tight_layout()

f, ax = plt.subplots(figsize=(10, 8))

corr = df_playlist_injury.corr()

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
df_playlist_injury.columns
df_playlist_injury_playtrack=pd.merge(df_playlist_injury,df_player_track,left_on='PlayKey',right_on='PlayKey',how="inner")
df_playlist_injury_playtrack.shape
df_playlist_injury_playtrack.head()
table = pd.pivot_table(df_playlist_injury_playtrack, values=['PlayerKey','s'], index=['Surface','BodyPart'], aggfunc={'PlayerKey': lambda x: x.count(),'s': [min, max, np.mean]} )

table_2 = pd.pivot_table(df_playlist_injury_playtrack, values= ['PlayerKey','s'], index=['Surface'],columns='BodyPart', aggfunc={'PlayerKey': lambda x: x.count(),'s': [min, max, np.mean]} )

table.unstack().plot(kind='bar', stacked=True,colormap="Blues",edgecolor ='black')

table.unstack()

plt.title('Speed per player that has and injury Per Type of Field')

plt.xlabel('Type of Field')

plt.ylabel('No. Players')

plt.legend( ('Ankle', 'Foot','Knee'))

plt.grid()

print(table)
data = df_playlist_injury_playtrack.groupby(['Surface','BodyPart'])['s'].agg({'LowValue':'min','HighValue':'max','Mean':'mean','Median':'median'})

data.reset_index().plot(x='BodyPart',kind='bar',colormap='Blues')

plt.title('Statistics Speed per injury  in each kind of field')

print(data)
data = df_playlist_injury_playtrack.groupby(['Surface','BodyPart'])['dis'].agg({'LowValue':'min','HighValue':'max','Mean':'mean','Median':'median'})

data.reset_index().plot(x='BodyPart',kind='bar',colormap='Blues')

plt.title('Statistics Distance per injury in each kind of field')

print(data)
df_playlist_injury_playtrack['PlayerKey'].nunique()
df_playlist_injury_playtrack['a']=(max(df_playlist_injury_playtrack['s']) - min(df_playlist_injury_playtrack['s'])  /  (max(df_playlist_injury_playtrack['time'])  - min(df_playlist_injury_playtrack['time'])  ) )
data = df_playlist_injury_playtrack.groupby(['Surface','BodyPart'])['a'].agg({'LowValue':'min','HighValue':'max','Mean':'mean','Median':'median'})

data.reset_index().plot(x='BodyPart',kind='bar',colormap='Blues')

plt.title('Statistics Aceleration per injury in each kind of field')

print(data)
df_playlist_injury_playtrack['da']=(max(df_playlist_injury_playtrack['s']) * max(df_playlist_injury_playtrack['s'])   - min(df_playlist_injury_playtrack['s']) * min(df_playlist_injury_playtrack['s']) ) /  (2 * sum(df_playlist_injury_playtrack['dis']) )
data = df_playlist_injury_playtrack.groupby(['Surface','BodyPart'])['da'].agg({'LowValue':'min','HighValue':'max','Mean':'mean','Median':'median'})

data.reset_index().plot(x='BodyPart',kind='bar',colormap='Blues')

plt.title('Statistics Decelaration per injury in each kind of field')

print(data)