import pandas as pd
import matplotlib.pyplot as plt
csv_pop_label = 'TotalPop' # from the csv file.
csv_pov_label = 'Poverty' # from csv file below. this is the property being investigated.

# We shall investigate Poverty wrt the following csv file columns for race: these could be automatically synthesized
# but this list is sort of ordered in decreasing population share just to make overlapping graphs easier, if used..

csv_race_labels= ["White", "Hispanic", "Black", "Asian", "Native", "Pacific"] # these are from the csv file.
# above labels can be automatically drawn from CSV file. But manually listing it shows its respective line color below:
line_colors = dict(zip(csv_race_labels, ['lightblue', 'green', 'chocolate', 'orchid', 'red', 'olive']))
csv_puerto_rico_label = 'Puerto Rico' # label from csv file. This analysis is for 50 states, PR is filtered out.
csv_dc_label = 'District of Columbia' # label from csv file. This analysis is for 50 states, DC is filtered out.

def cumulative_column(xcol):
    """To draw out the percentile information of a column listed by %age"""
    xres = []
    xval = 0
    for itm in xcol:
        xval += itm # because xcol is a percentage.
        xres.append(xval)
    return xres
def comprs(X, Freq):
    """compresses consecutive values (x1,freq1) (x2,freq2) .. to (x1, freq1+freq2+..) when x1 == x2 = ..
       input: X is a list of values, while freq is the matching list."""
    res = []
    res_freq = []
    if X is None:
        return (res, res_freq)
    val = X[0]
    total = 0
    for itm, ct in zip(X, Freq):
        if itm == val:
            total = total + ct
        else:
            res.append(val)
            res_freq.append(total)
            val = itm
            total = ct
    if(res[-1] != val):
        res.append(val)
        res_freq.append(total)
    return (res, res_freq)
def misc_prep_50_states_data(info):
    """remove rows with zero population, and also Puerto Rico. remove rows with no poverty information"""
    op = info.agg(csv_pop_label).sum()
    print("checking", len(info), "rows .., Population:", op)
    missing_info_col = info.columns[info.isna().any()].tolist()
    if csv_pov_label in missing_info_col:
        ol = len(info)
        info = info[~info[csv_pov_label].isnull()]
        np = info.agg(csv_pop_label).sum()
        dp = round((op - np) / op * 100, 3)
        print(csv_pov_label,  " -- missing info in", ol - len(info), "rows.", "Population reduced in calculation:", dp, '%', "New population:", np)
        op = info.agg(csv_pop_label).sum()    
    if csv_puerto_rico_label in set(info['State'].values):
        ol = len(info)
        info=info[(info['State'] != csv_puerto_rico_label)] # this could be done via any exclusion list -- out of 50 states -- but is simpler.
        np = info.agg(csv_pop_label).sum()
        dp = round((op - np) / op * 100, 3)
        print(csv_puerto_rico_label, " -- ", ol - len(info), "rows removed.", "Population reduced in calculation:", dp, "%", "New population:", np)
        #print("----", info[csv_pop_label].sum())

        op = info.agg(csv_pop_label).sum()    
    if csv_dc_label in set(info['State'].values):
        ol = len(info)
        info=info[(info['State'] != csv_dc_label)] # this could be done via any exclusion list -- out of 50 states -- but is simpler.
        np = info.agg(csv_pop_label).sum()
        dp = round((op - np) / op * 100, 3)
        print(csv_dc_label, " -- ", ol - len(info), "rows removed.", "Population reduced in calculation:", dp, "%", "New population:", np)
    # sort the tables by csv_pov_label property = 'poverty' here. we are going to use percentiles.
    print("Sort table by its '"+ csv_pov_label +"' column")
    info=info.sort_values(by=csv_pov_label, ascending=True)
    return info
def plot_cycle(info, state="USA 50 states", histogram = None):
        r_pop_percent = {}  # population per racial group percentage.
        r_pop_num = {}      # actual number of people in a racial group
        r_candidates= []
        r_wtd_pov = {}      # weighted poverty for each race
        pie_pct= []          # subset
        drop_percentage = 1 # we drop any group whose population is less than this value. 
        dropped = False

        tinfo = info.copy()  # this copy is un-necessary.  but for someone running this script cell multiple times,
        # tinfo % columns'll be converted multiple times in the next steps ..
        pop = tinfo.agg(csv_pop_label).sum()
        print("Total database: entries x columns:", tinfo.shape, "-- Total population#:", pop)

        dec_race = 0
        for r in csv_race_labels:
            tinfo[r] = tinfo[csv_pop_label] * tinfo[r] / 100.0  # convert per row percent table to actual numbers.
            r_pop_num[r] = tinfo.agg(r).sum() # race total number of people. (should round it the way it is calculated, but not needed..)       
            perct = round( r_pop_num[r] / pop * 100.0, 1)
            r_pop_percent[r] = perct  # race total as a percent of total us population
            dec_race = dec_race + perct
            if perct < drop_percentage:
                print(r, ':', perct, "% -- dropped")
                dropped = True
            else:
                print(r, ':', perct, '%')
                r_candidates.append(r)
                pie_pct.append(perct)

        if dropped:
            print("(Population group <", drop_percentage, "% is not used here, for margin of error & to ease visualization.)")
        print("Undeclared race:", round(100 - dec_race, 1), "%")
        lbl = ["%s: %.1f%%"%(r,p) for r,p in zip(r_candidates, pie_pct)]
        col = [line_colors[r] for r in r_candidates]
        if((100 - dec_race) > 1):  # show undeclared race wedge in the pie chart if greater than 1%
            pct = round(100 - dec_race, 1)
            pie_pct.append(pct)
            lbl.append("undec: " + str(pct) + "%")
            col.append("white")
        plt.pie(pie_pct, labels=lbl, colors = col)
        plt.axis('equal')
        plt.title('Census 2015: ' + state + ' population')
        plt.show()

        ### plot charts:

        f = plt.figure(figsize=[12,10])
        plt.axis([0,75, 0,100]) # x scale: 0 to 75% poverty. y = 0 to 100% population.
        #f = plt.figure()
        plt.title("poverty")
        ax = f.add_axes([0,0,1,1])
        #ax = f.add_subplot(1,1,1, aspect=2)
        ax.set_xlabel("Poverty %")
        ax.set_ylabel("Percentile distribution (%) within each group")
        ax.set_ylim([0, 100])
        ax.set_autoscalex_on(False)
        ax.set_xlim([0,75])
        plt_spc_idx = 10
        plt_spc_gap = 5
        plt_lbl_x = 40
        plt_lbl_y = 30

        for race in r_candidates:
            (x,freq) = comprs(tinfo[csv_pov_label].values, tinfo[race].values)
            cfreq = cumulative_column(freq)
            y = [itm / r_pop_num[race] * 100 for itm in cfreq] # make percentile. normalize.
            line, = plt.plot(x, y, '-')
            plt.setp(line, color=line_colors[race])
            r_wtd_pov[race] =  (tinfo[race] * tinfo[csv_pov_label]).sum() / r_pop_num[race]
            plt.text(plt_lbl_x, plt_lbl_y, race + ": " + str(round(r_wtd_pov[race], 1)) + "%", color=line_colors[race])
            plt_lbl_y = plt_lbl_y + plt_spc_gap
            print(race, "weighted poverty:", round(r_wtd_pov[race], 1))
        r_wtd_pov['diff'] = round(max(r_wtd_pov.values()) - min(r_wtd_pov.values()),2)
        r_wtd_pov['all'] = (tinfo[csv_pop_label] * tinfo[csv_pov_label]).sum() / pop
        r_wtd_pov['name'] = state
        print("All weighted poverty:", round(r_wtd_pov['all'], 1))
        
        (x,freq) = comprs(tinfo[csv_pov_label].values, tinfo[csv_pop_label].values)
        cfreq = cumulative_column(freq)
        y = [itm / pop * 100 for itm in cfreq]
        line, = plt.plot(x, y, '--')
        plt.setp(line, color='black')
        plt.text(plt_lbl_x, plt_lbl_y, state + ' (total) --- ' + str(round(r_wtd_pov['all'], 1)) + "% in poverty" , color='black')
        plt.show()

        if histogram is None:
            return r_wtd_pov
        
        (x,freq) = comprs(tinfo[csv_pov_label].values, tinfo[csv_pop_label].values)
        #y = [itm / pop * 100 for itm in freq]
        plt.bar(x, freq, color='black')
        plt.title(state + ' : ' + csv_pov_label)
        plt.ylabel(state + ": population")
        plt.show()
        
        for race in r_candidates:
            (x,freq) = comprs(tinfo[csv_pov_label].values, tinfo[race].values)
            #y = [itm / r_pop_num[race] * 100 for itm in freq]
            plt.bar(x, freq, color=line_colors[race])
            plt.title(state + ' -- ' + race + ': ' + csv_pov_label)
            plt.ylabel(race + " population")
            plt.show()

        return r_wtd_pov

oinfo=pd.read_csv('../input/acs2015_census_tract_data.csv')
# change this path if the file is different or at a different location
oinfo = misc_prep_50_states_data(oinfo)
print("...USA 50 states...")
wt_usa = plot_cycle(oinfo)
print("Poverty in America, 2015:----")
l = [(i[0], round(i[1],1)) if i[0] in csv_race_labels else ('', 0) for i in sorted(wt_usa.items(), key=lambda x:round(x[1], 2) if x[0] in csv_race_labels else 100, reverse=True)]
l1,l2 = zip(*l)
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.barh(l1,l2, 0.30)
ax.set_title("USA 2015: weighted poverty %")
plt.show()
#per state trends.. 

wt_pov = {}  # weighted poverty information is collected here per state of USA.
states = sorted(set(oinfo['State'].values))
for st in states:
    print("Analysing", st, "..")
    wt_pov[st] = plot_cycle(oinfo[oinfo['State'] == st], st, histogram = None)

N = 50  # change this number if lesser number, e.g. top 5 to be shown. Following bar charts show all (N=50) if possible.
# the plots below show bar charts, which are customized mostly for captioning or size of the bar chart. Else they could be in a loop..
race = "Black"
print(race+'s' + " in America :----")
print("Also see: https://www.theroot.com/the-5-best-states-for-black-people-1790877760")
l = [(i[0], round(i[1][race],1)) if race in i[1] else ('', 0) for i in sorted(wt_pov.items(), key=lambda x:round(x[1][race], 2) if race in x[1] else 100, reverse=True)[0:N]]
l1,l2 = zip(*l)
fig = plt.figure(figsize=(10,12))
ax = fig.add_subplot(111)
ax.barh(l1,l2, 0.30)
ax.set_title(race + 's' + " in America: weighted poverty %, per state")
plt.show()
race = "Hispanic"
print(race+'s' + " in America :----")
l = [(i[0], round(i[1][race],1)) if race in i[1] else ('', 0) for i in sorted(wt_pov.items(), key=lambda x:round(x[1][race], 2) if race in x[1] else 100, reverse=True)[0:N]]
l1,l2 = zip(*l)
fig = plt.figure(figsize=(10,12))
ax = fig.add_subplot(111)
ax.barh(l1,l2, 0.30)
ax.set_title(race + 's' + " in America: weighted poverty %, per state")
plt.show()
race = "White"
print(race+'s' + " in America :----")
l = [(i[0], round(i[1][race],1)) if race in i[1] else ('', 0) for i in sorted(wt_pov.items(), key=lambda x:round(x[1][race], 2) if race in x[1] else 100, reverse=True)[0:N]]
l1,l2 = zip(*l)
fig = plt.figure(figsize=(10,12))
ax = fig.add_subplot(111)
ax.barh(l1,l2, 0.30)
ax.set_title(race + 's' + " in America: weighted poverty %, per state")
plt.show()
race = "Asian"
print(race+'s' + " in America :----")
l = [(i[0], round(i[1][race],1)) if race in i[1] else ('', 0) for i in sorted(wt_pov.items(), key=lambda x:round(x[1][race], 2) if race in x[1] else 100, reverse=True)[0:N]]
l1,l2 = zip(*l)
fig = plt.figure(figsize=(10,12))
ax = fig.add_subplot(111)
ax.barh(l1,l2, 0.30)
ax.set_title(race + 's' + " in America: weighted poverty %, per state")
plt.show()
race = "Native"
print(race+'s' + " in America :----")
l = [(i[0], round(i[1][race],1)) if race in i[1] else ('', 0) for i in sorted(wt_pov.items(), key=lambda x:round(x[1][race], 2) if race in x[1] else 100, reverse=True)[0:N]]
l1,l2 = zip(*l)
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.barh(l1,l2, 0.30)
ax.set_title(race + 's' + " in America: weighted poverty %, per state")
plt.show()
print("It is worth looking at the last two states individual graphs, to see the plight of Natives!")
race = "Pacific"
print(race+'s' + " in America :----")
l = [(i[0], round(i[1][race],1)) if race in i[1] else ('', 0) for i in sorted(wt_pov.items(), key=lambda x:round(x[1][race], 2) if race in x[1] else 100, reverse=True)[0:N]]
l1,l2 = zip(*l)
fig = plt.figure(figsize=(10,2))
ax = fig.add_subplot(111)
ax.barh(l1,l2, 0.10)
ax.set_title(race + 's' + " in America: weighted poverty %, per state")
plt.show()
print(N, "States & maximum differntial in poverty (best first) :----")
l = [(i[0], round(i[1]['diff'])) for i in sorted(wt_pov.items(), key=lambda x:x[1]['diff'], reverse=True)[0:N]]
l1,l2 = zip(*l)
fig = plt.figure(figsize=(10,12))
ax = fig.add_subplot(111)
ax.barh(l1,l2, 0.30)
ax.set_title("States & poverty differential % (i.e. inequality) amongst its races:")
plt.show()
print(N, "States & average poverty (best first) :----")
l = [(i[0], round(i[1]['all'])) for i in sorted(wt_pov.items(), key=lambda x:x[1]['all'], reverse=True)[0:N]]
l1,l2 = zip(*l)
fig = plt.figure(figsize=(10,12))
ax = fig.add_subplot(111)
ax.barh(l1,l2, 0.30)
ax.set_title("States & average poverty (weighted) %")
plt.show()
print("Vermont: an amazing trendline for Hispanics, compared to most other states! https://www.burlingtonfreepress.com/story/news/2014/02/20/vermont-immigration-trends-differ-dramatically-from-us-picture/5607037/")
print("See: Blacks in Wisconsin: http://dollarsandsense.org/archives/2015/1115schneider.html")