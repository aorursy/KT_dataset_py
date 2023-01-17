import os
print(os.listdir("../input"))
import pandas as pd
df = pd.read_csv("../input/loan.csv", dtype="object")
temp = df.drop(labels=['id','member_id','hardship_payoff_balance_amount','hardship_last_payment_amount','debt_settlement_flag_date','settlement_status','settlement_date','payment_plan_start_date','hardship_length','hardship_dpd','hardship_loan_status','orig_projected_additional_accrued_interest','disbursement_method','debt_settlement_flag','settlement_amount','settlement_percentage','settlement_term'],axis=1)
temp
