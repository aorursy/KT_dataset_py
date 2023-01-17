import pandas as pd

disrupted_diseases_df = pd.read_csv('../input/disruptions-by-diseases-worldwide/DisruptionByDisease.csv')

print(disrupted_diseases_df.head)

disrupted_diseases_df.plot.bar(x='Disease', color=['b','r'])
disrupted_phases_df = pd.read_csv('../input/disruption-by-covid19-transmission-phases/DisruptionByCOVID19TransmissionPhases.csv')

print(disrupted_phases_df.head)

disrupted_phases_df.plot.bar(x='Disease', color=['b','y','r'])
disruption_causes_df = pd.read_csv('../input/causes-of-disruption/CausesOfDisruption.csv')

print(disruption_causes_df.head)

disruption_causes_df.plot.bar(x='Cause', color=['r'])
vulnerability_types_df = pd.read_csv('../input/vulnerability-of-ncd-with-covid19/VulnerabilityOfNCDWithCOVID19.csv')

print(vulnerability_types_df.head)

vulnerability_types_df.plot.bar(x='Vulnerability type', color=['r'])
trend_shifts_df = pd.read_csv('../input/trend-shifting-in-caring-ncd/TrendShiftingInCare.csv')

print(trend_shifts_df.head)

trend_shifts_df.plot.bar(x='Approach', color=['b'])