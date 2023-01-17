f = open("../input/submission/final_5_0.10117320406891336.csv")
tem = pd.read_csv(f)
tem.head(10)
tem.to_csv('submission.csv', index=False)