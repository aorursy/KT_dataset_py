from babel.numbers import format_currency

from requests import get

from zipfile import ZipFile

from csv import DictReader

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

from pprint import pprint



def retrieve_open_data_parliament_budget_file():

    """

        Get the Zip archive of parliament budget for last mandature (2012-2017)

    """

    response = get('http://data.assemblee-nationale.fr/static/openData/repository/BudgetsAN/BudgetsAN_2012-2017.csv.zip', stream=True)



    with open('files/archives/parliament_budget.zip', 'wb') as archive:

        for chunk in response.iter_content(chunk_size=128):

            archive.write(chunk)



def unzip_data_parliament_budget_file():

    """

        Extract the Zip archive into the files/data folder

    """

    with ZipFile('files/archives/parliament_budget.zip', 'r') as zip_item:

        zip_infos = zip_item.infolist()

        for zip_info in zip_infos:

            zip_info.filename = 'parliament_budget.csv'

        zip_item.extract(zip_info, 'files/data/')



def retrieve_open_data_national_budget_file():

    """

        Get the France national budget for 2019

    """

    response = get('https://www.data.gouv.fr/fr/datasets/r/19eb1f46-8d68-4b57-934a-d84ad00942f6', stream=True)



    with open('files/data/national_budget.csv', 'wb') as archive:

        for chunk in response.iter_content(chunk_size=128):

            archive.write(chunk)



def collect_data():

    """

        Meta function used to collect all the required data

    """

    retrieve_open_data_parliament_budget_file()

    unzip_data_parliament_budget_file()

    retrieve_open_data_national_budget_file()



### Collecting the Data ... bip bip bip ...

#collect_data()
def get_parliament_budget_entry(row_label):

    """

        Extract information from parliament budget CVS file

        into a Python dictionnary

    """

    with open('../input/costofdeputyday/parliament_budget.csv', encoding="utf-8") as budget_file:

        budget_reader = DictReader(budget_file, delimiter=';', quotechar="\"")

        budget = []

        for budget_entry in budget_reader:

            budget.append((budget_entry['\ufeff'], budget_entry['Libellé'], budget_entry[row_label]))

    

    return budget



def get_national_budget_entry():

    """

        Extract information from the national budget CSV file

        into a Python dictionnary

    """

    with open('../input/costofdeputyday/national_budget.csv', encoding="utf-8") as budget_file:

        budget_reader = DictReader(budget_file, delimiter=';', quotechar="\"")

        budget = []

        for budget_entry in budget_reader:

            budget.append((budget_entry['Programme'], int(budget_entry['CP (T2 + HORS T2)  LFI 2019'].replace(" ", ""))))

    

    return budget
def get_first_deputies_cost_per_day(budget):

    """

        We consider that the cost of a day of work is exactly

        the budget of a year divided by the number of days



        i.e cost_day = cost_year / 365



        => 1 578 891 €

    """

    for budget_entry in budget:

        if budget_entry[1] == ' DEPENSES TOTALES':

            cost_a_day = int(budget_entry[2].replace(" ", "")) / 365



            return cost_a_day
def get_second_deputies_cost_per_day(budget):

    """

        We consider that the cost of a day of work is exactly

        the budget of parliament wage costs divided by the number of days



        i.e cost_day = cost_wage_costs_year / 365



        => 1 375 497 €

    """

    cost_a_day = 0

    for budget_entry in budget:

        if budget_entry[1] == 'CHARGES DE PERSONNEL':

            cost_a_day = cost_a_day + int(budget_entry[2].replace(" ", ""))



        if budget_entry[1] == 'CHARGES PARLEMENTAIRES':

            cost_a_day = cost_a_day + int(budget_entry[2].replace(" ", ""))



    return cost_a_day / 365
def get_part_of_wages_in_total_budget(budget):

    """

        We calculate the part of wages in the total budget



        => ~ 87,12 %

    """

    cost = 0

    for budget_entry in budget:

        if budget_entry[1] == 'CHARGES DE PERSONNEL':

            cost = cost + int(budget_entry[2].replace(" ", ""))



        if budget_entry[1] == 'CHARGES PARLEMENTAIRES':

            cost = cost + int(budget_entry[2].replace(" ", ""))



        if budget_entry[1] == ' DEPENSES TOTALES':

            part = 100 * cost / int(budget_entry[2].replace(" ", ""))

    

    return part



def get_third_deputies_cost_per_day(budget):

    """

        We don't have the details of budget for 2020 (not published, yet)

        But we could say it's ... the same % of the budget than for 2017

        The total budget is published, see http://www.assemblee-nationale.fr/dyn/15/textes/l15b2272_projet-loi#_Toc20502982



        => 1 236 096 €

    """

    # 517 890 000€

    total_budget_2020 = 517890000



    return (get_part_of_wages_in_total_budget(budget)/100 * total_budget_2020) / 365
def format_budget(x, pos):

    """

        Format integer values into prices in Euros

    """

    return format_currency(x, 'EUR', locale = 'fr')



def display_cost_a_day_bar_chart(budget):

    """

        Display a bar chart comparing the cost of a day

        of Parliament taking in account 3 hypotheses

    """

    figure, axes = plt.subplots()

 

    plt.bar(

        ['Hypothèse 1', 'Hypothèse 2', 'Hypothèse 3'],

        [

            get_first_deputies_cost_per_day(budget),

            get_second_deputies_cost_per_day(budget),

            get_third_deputies_cost_per_day(budget)

        ]

        )

    plt.title("Coût d'une journée de travail parlementaire")

    formatter = FuncFormatter(format_budget)

    axes.yaxis.set_major_formatter(formatter) 

    plt.show()







parliament_budget = get_parliament_budget_entry('Réalisé 2017')

display_cost_a_day_bar_chart(parliament_budget)
def display_national_budget(budget):

    """

        Display a pie chart of national budget

    """

    figure, axes = plt.subplots()

    

    national_budget = 0

    parliament_budget = 0



    for budget_entry in budget:

        if budget_entry[0] == 'Assemblée nationale':

            parliament_budget = budget_entry[1]

        national_budget = national_budget + budget_entry[1]



    budgets = [national_budget, parliament_budget]

    explode = [0.1, 0.5]

    labels = ["Total", "Assemblée nationale"]

    axes.axis("equal")

    axes.pie(budgets,

        labels = labels,

        autopct="%1.1f%%",

        explode=explode)

    plt.title("Répartition du budget de l'Etat (2019)")

    plt.show()



def display_national_funds():

    """

        Display a pie chart of sources of national budget

        See https://www.performance-publique.budget.gouv.fr/sites/performance_publique/files/files/documents/ressources_documentaires/documentation_budgetaire/chiffres_cles/LFI2019-budget-chiffres-cles.pdf (page 5)

    """

    figure, axes = plt.subplots()

    

    national_budget = 0

    parliament_budget = 0



    budgets = [409.415, 70.426]

    labels = ["Total recettes fiscal (brut)", "Impôt sur le revenu"]

    axes.axis("equal")

    axes.pie(budgets,

        labels = labels,

        autopct="%1.1f%%"

    )

    plt.title("Répartition des recettes de l'Etat l'Etat (2019)")

    plt.show()



national_budget = get_national_budget_entry()

display_national_budget(national_budget)

display_national_funds()