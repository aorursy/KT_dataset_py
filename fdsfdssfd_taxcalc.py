# %%
from enum import Enum

WORK_DAYS_WITH_VAC = 253 - 20


class IncomeType(Enum):
    SALARY = 0
    DIVIDEND = 1
    BOARD_BONUS = 5
    CONTRACT_ROYALTIES = 2
    OTHER_INCOME = 3
    SOLE_PROPRIETORSHIP = 4
    PATENT_PERMIT = 6
    PARTNERSHIP_SALARY = 7
    FARMER_EDV_4 = 15


class Social(Enum):
    SALARY = 0
    CONTRACT_ROYALTIES = 1
    OTHER_INCOME = 2
    SOLE_PROPRIETORSHIP = 3
    PARTNERSHIP = 4
    PATENT_PERMIT = 8
    FARMING_COM_INCOME_TAX = 5
    FARMING_COM_NO_TAX = 6
    FARMER_EDV_4 = 7
    UNINSURED = 10
    BOARD_BONUS = 11


PatentCostsPerDay = 2.3


# %%
def calculate_income_tax(yearly_income: float, type: IncomeType):
    main_rate = None
    cutoff = None
    cutoff_rate = None
    total_tax = None

    if type == IncomeType.SALARY:
        main_rate = 0.2
        cutoff_rate = 0.32
        cutoff = 104277.60

    elif type == IncomeType.FARMER_EDV_4:
        main_rate = 0.0
        cutoff_rate = 0.15
        cutoff = 45000


    elif type == IncomeType.PARTNERSHIP_SALARY:
        main_rate = 0.2
        cutoff_rate = 0.32
        cutoff = 104277.60

    elif type == IncomeType.DIVIDEND:
        main_rate = 0.15

    elif type == IncomeType.CONTRACT_ROYALTIES or type == IncomeType.BOARD_BONUS:
        main_rate = 0.2
        cutoff_rate = 0.32
        cutoff = 104277.60

    elif type == IncomeType.OTHER_INCOME:

        main_rate = 0.15
        cutoff_rate = 0.2

        cutoff = 148968

    elif type == IncomeType.PATENT_PERMIT:

        main_rate = 0.0
        cutoff_rate = 0.15

        cutoff = 45000

    elif type == IncomeType.SOLE_PROPRIETORSHIP:
        main_rate = 0.15

    if type == IncomeType.SOLE_PROPRIETORSHIP:

        if yearly_income <= 20000:
            credit = yearly_income * 0.1
        elif yearly_income > 20000 and yearly_income < 35000:
            # credit = yearly_income * (0.1-2/(300000 * (yearly_income - 20000)))
            credit = yearly_income * (0.1 - 2 / 300000 * (yearly_income - 20000))
        else:
            credit = 0;

        total_tax = yearly_income * main_rate - credit

    elif type == IncomeType.SALARY:
        if yearly_income < 24705:
            if (yearly_income <= 4200):
                total_tax = 0
            else:
                if (yearly_income <= 7284):
                    untaxed_income = 4200
                else:
                    untaxed_income = 4200 - 0.17 * (yearly_income - 7284)
                    untaxed_income = untaxed_income if untaxed_income > 0 else 0
                total_tax = (yearly_income - untaxed_income) * main_rate

    if total_tax == None:
        if cutoff is None or yearly_income < cutoff:
            total_tax = yearly_income * main_rate
        else:
            total_tax = cutoff * main_rate + (yearly_income - cutoff) * cutoff_rate

    return total_tax, yearly_income - total_tax, (1 / yearly_income * total_tax)


def calcutate_social_tax(yearly_income: float, type: Social):
    total_tax = 0
    cutoff = 84 * 1241.40

    if type == Social.SALARY:
        # total_tax = yearly_income * 0.195

        if (yearly_income > cutoff):
            yearly_income = cutoff

        total_tax = yearly_income * 0.195


    elif type == Social.CONTRACT_ROYALTIES:

        if (yearly_income > cutoff):
            yearly_income = cutoff

        total_tax = yearly_income * 0.195


    elif type == Social.BOARD_BONUS:
        if (yearly_income > cutoff):
            yearly_income = cutoff

        total_tax = yearly_income * 0.157

    elif type == Social.OTHER_INCOME:
        cutoff = 43 * 1241.40

        if (yearly_income > cutoff):
            yearly_income = cutoff

        total_tax = (yearly_income * 0.5) * 0.195

    elif type == Social.SOLE_PROPRIETORSHIP or type == Social.FARMING_COM_INCOME_TAX:

        cutoff = 43 * 1241.40

        if (yearly_income > cutoff):
            yearly_income = cutoff

        total_tax = (yearly_income * 0.9) * 0.195

    elif type == Social.PARTNERSHIP:

        cutoff = 43 * 1241.40

        if (yearly_income > cutoff):
            yearly_income = cutoff

        total_tax = (yearly_income * 0.5) * 0.2081

    elif type == Social.PATENT_PERMIT:
        total_tax = (607 * 12 * 0.157) * 0.875

    elif type == Social.UNINSURED:
        total_tax = 607 * 0.698

    elif type == Social.FARMER_EDV_4:

        health_tax = (yearly_income * 0.9) * 0.0698
        if (health_tax > 3725):
            health_tax = 3725

        social_tax = (yearly_income * 0.9) * 0.125
        if (social_tax > 6683):
            social_tax = 6683

        total_tax = health_tax + social_tax

    return total_tax, yearly_income - total_tax, (1 / yearly_income * total_tax)


# %% Test income calc

def calculate_salary(income):
    income_tax = calculate_income_tax(income, IncomeType.SALARY)[0]
    social_tax = calcutate_social_tax(income, Social.SALARY)[0]

    return income - income_tax - social_tax, (1 / income * (income_tax + social_tax))


def calculate_propertorship_30(income):
    income_tax = calculate_income_tax(income * 0.7, IncomeType.SOLE_PROPRIETORSHIP)[0]
    social_tax = calcutate_social_tax(income * 0.7, Social.SOLE_PROPRIETORSHIP)[0]

    return income - income_tax - social_tax, (1 / income * (income_tax + social_tax))


def calculate_propertorship_exp(income):
    income_tax = calculate_income_tax(income, IncomeType.SOLE_PROPRIETORSHIP)[0]
    social_tax = calcutate_social_tax(income, Social.SOLE_PROPRIETORSHIP)[0]

    return income - income_tax - social_tax, (1 / income * (income_tax + social_tax))


def calculate_partnership(income):
    income_tax = calculate_income_tax(income, IncomeType.PARTNERSHIP_SALARY)[0]
    social_tax = calcutate_social_tax(income, Social.PARTNERSHIP)[0]

    return income - income_tax - social_tax, (1 / income * (income_tax + social_tax))


def calculate_dividend(income):
    income_tax = calculate_income_tax(income, IncomeType.DIVIDEND)[0]
    social_tax = calcutate_social_tax(income, Social.UNINSURED)[0]
    return income - income_tax - social_tax, (1 / income * (income_tax + social_tax))


def calculate_patent(income: float):
    fixed_cost = WORK_DAYS_WITH_VAC * PatentCostsPerDay

    income_tax = calculate_income_tax(income, IncomeType.PATENT_PERMIT)[0]
    social_tax = calcutate_social_tax(income, Social.PATENT_PERMIT)[0]

    total_tax = income_tax + social_tax + fixed_cost

    return income - total_tax, (1 / income * total_tax)


def calculate_royalties(income: float):
    income_tax = calculate_income_tax(income, IncomeType.OTHER_INCOME)[0]
    social_tax = calcutate_social_tax(income, Social.OTHER_INCOME)[0]
    return income - income_tax - social_tax, (1 / income * (income_tax + social_tax))


def calculate_contract_royalties(income: float):
    income_tax = calculate_income_tax(income, IncomeType.CONTRACT_ROYALTIES)[0]
    social_tax = calcutate_social_tax(income, Social.CONTRACT_ROYALTIES)[0]
    return income - income_tax - social_tax, (1 / income * (income_tax + social_tax))


def calculate_royalties(income: float):
    income_tax = calculate_income_tax(income, IncomeType.OTHER_INCOME)[0]
    social_tax = calcutate_social_tax(income, Social.OTHER_INCOME)[0]
    return income - income_tax - social_tax, (1 / income * (income_tax + social_tax))


def calculate_board_bonuses(income: float):
    income_tax = calculate_income_tax(income, IncomeType.BOARD_BONUS)[0]
    social_tax = calcutate_social_tax(income, Social.BOARD_BONUS)[0]
    return income - income_tax - social_tax, (1 / income * (income_tax + social_tax))


def calculate_farmer(income: float):
    income_tax = calculate_income_tax(income, IncomeType.FARMER_EDV_4)[0]
    social_tax = calcutate_social_tax(income, Social.FARMER_EDV_4)[0]
    return income - income_tax - social_tax, (1 / income * (income_tax + social_tax))


# %% Example calculate and graph renting property in Vilnius
# Test data based on https://www.tax.lt/skaiciuokles/atlyginimo_ir_mokesciu_skaiciuokle (it does not account for the progressive income tax bracket and the contribution cap.

salary_test_data = [
    {"gross": 100 * 12, "net": 80.5},
    {"gross": 600 * 12, "net": 443},
    {"gross": 1000 * 12, "net": 670.07},
    {"gross": 1500 * 12, "net": 953.57},
    {"gross": 2000 * 12, "net": 1237.07},
    {"gross": 3000 * 12, "net": 1815.00 * 12},
    {"gross": 4000 * 12, "net": 2420 * 12},
    {"gross": 5000 * 12, "net": 3025.00 * 12},
    {"gross": 10000 * 12, "net": 6050.00 * 12},
    {"gross": 15000 * 12, "net": 9075.00 * 12},
    {"gross": 50000 * 12, "net": 30250.00 * 12},
    {"gross": 100000 * 12, "net": 60500.00 * 12},
]

proprietorship_test_data = [
    {"gross": 100 * 12, "net": 1010},
    {"gross": 600 * 12, "net": 6063.4},
    {"gross": 1000 * 12, "net": 10105.8},
    {"gross": 50000, "net": 38607},
    {"gross": 100000, "net": 79090},
    {"gross": 1000000, "net": 884590.86},
]


def calculate_tax_rate(income, func):
    total_rem = func(income)[0]

    return 1 - ((1 / income) * total_rem), total_rem;


def run_test(test_data, func):
    for row in test_data:
        gross, exp_net = int(row["gross"]), int(row["net"])

        exp = int(func(gross)[0])

        print("Gross : {0:<10}  net : {1:<10}({4:.2f})   expected : {2:<10}({5:.2f}) diff {3:<10}".format(
            gross,
            exp,
            exp_net,
            exp - exp_net,

            1 - 1 / gross * exp,
            1 - 1 / gross * exp_net
        )
        )


print("\nSalary:")
run_test(salary_test_data, calculate_salary)

print("\nProprietorship:")
run_test(proprietorship_test_data, calculate_propertorship_30)

# %% Build DataFrame

income_types = {
    "Salary": calculate_salary,
    "tax_SoleProprietorshipExp": calculate_propertorship_exp,
    "Dividend": calculate_dividend,
    "PermitHairdresserOrConstructionWorker": calculate_patent,
    "Royalties": calculate_royalties,
    "BoardBonuses": calculate_board_bonuses,
    "Agriculture": calculate_farmer,
}

x_points = [
    # 1000,
    # 1000,
    3000,
    4500,
    7284,  # Min Wage
    8500,
    10000,
    12000,
    14000,
    16296,  # Avg Wage
    20000,
    24000,
    28000,
    32000,  #
    38000,
    44000,
    50000,
    56000,
    62000,
    68000,
    80000,  #
    100000,  #
    125000,  #
    150000,
    175000,
    200000,  #
    250000,  #
    300000,  #
    350000,  #
    400000,  #
    450000,  #
    500000,
    600000,
    700000,
    800000,
    900000,
    1000000
]

columns = {}

# columns = {k: [] for (k, v) in income_types.items()}

for col in income_types.keys():
    columns[F"tax_{col}"] = []
    # columns[F"inc_{col}"] = []

columns["income"] = []
for x in x_points:

    columns["income"].append(x)
    for col in income_types.keys():
        r = calculate_tax_rate(x, income_types[col])
        columns[F"tax_{col}"].append(r[0])
        # columns[F"inc_{col}"].append(r[1])

import pandas as pd

df = pd.DataFrame.from_dict(columns)
df = df.set_index("income")
df["income"] = df.index

df.to_csv("tax_data_sample_2.csv")

print("Wrote to csv")
# %% Wide to long
import pandas as pd

long = pd.wide_to_long(df, stubnames='tax', i="income", j="type", sep='_', suffix='\w+')

long.to_csv("tax_data_sample_long.csv")


