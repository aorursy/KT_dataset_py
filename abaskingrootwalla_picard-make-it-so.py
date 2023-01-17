import json
# What would you like to find?

line_to_match = "make it so"

character = "PICARD"

series = "TNG"
data = json.load(open('/kaggle/input/start-trek-scripts/all_series_lines.json', 'r'))
# Double check number of Episodes

print(len(data[series].keys()))
# Thats actually 2-less than the actual episode count, but that's caused by

# Encounter at Farpoint & All Good Things ... being combined.

# Two-part series than span two seasons are considered single episodes
matching_lines = []

tng_episodes = data[series].keys()

for ep in tng_episodes:

    script = data[series][ep]

    character_lines = script[character]

    

    for l in character_lines:

        if line_to_match in l.lower():

            matching_lines.append((ep, l))
episodes = {

1: "Encounter at Farpoint Part I",

2: "Encounter at Farpoint Part II",

3: "The Naked Now",

4: "Code of Honor",

5: "The Last Outpost",

6: "Where No One Has Gone Before",

7: "Lonely Among Us",

8: "Justice",

9: "The Battle",

10: "Hide and Q",

11: "Haven",

12: "The Big Goodbye",

13: "Datalore",

14: "Angel One",

15: "11001001",

16: "Too Short a Season",

17: "When the Bough Breaks",

18: "Home Soil",

19: "Coming of Age",

20: "Heart of Glory",

21: "The Arsenal of Freedom",

22: "Symbiosis",

23: "Skin of Evil",

24: "We'll Always Have Paris",

25: "Conspiracy",

26: "The Neutral Zone",

27: "The Child",

28: "Where Silence Has Lease",

29: "Elementary, Dear Data",

30: "The Outrageous Okona",

31: "Loud as a Whisper",

32: "The Schizoid Man",

33: "Unnatural Selection",

34: "A Matter of Honor",

35: "The Measure of a Man",

36: "The Dauphin",

37: "Contagion",

38: "The Royale",

39: "Time Squared",

40: "The Icarus Factor",

41: "Pen Pals",

42: "Q Who",

43: "Samaritan Snare",

44: "Up the Long Ladder",

45: "Manhunt",

46: "The Emissary",

47: "Peak Performance",

48: "Shades of Gray",

50: "Evolution", # Notice this is out of order

49: "The Ensigns of Command",

51: "The Survivors",

52: "Who Watches the Watchers",

53: "The Bonding",

54: "Booby Trap",

55: "The Enemy",

56: "The Price",

57: "The Vengeance Factor",

58: "The Defector",

59: "The Hunted",

60: "The High Ground",

61: "Déjà Q",

62: "A Matter of Perspective",

63: "Yesterday's Enterprise",

64: "The Offspring",

65: "Sins of the Father",

66: "Allegiance",

67: "Captain's Holiday",

68: "Tin Man",

69: "Hollow Pursuits",

70: "The Most Toys",

71: "Sarek",

72: "Ménage à Troi",

73: "Transfigurations",

74: "The Best of Both Worlds, Part I",

75: "The Best of Both Worlds, Part II",

76: "Family",

77: "Brothers",

78: "Suddenly Human",

79: "Remember Me",

80: "Legacy",

81: "Reunion",

82: "Future Imperfect",

83: "Final Mission",

84: "The Loss",

85: "Data's Day",

86: "The Wounded",

87: "Devil's Due",

88: "Clues",

89: "First Contact",

90: "Galaxy's Child",

91: "Night Terrors",

92: "Identity Crisis",

93: "The Nth Degree",

94: "Qpid",

95: "The Drumhead",

96: "Half a Life",

97: "The Host",

98: "The Mind's Eye",

99: "In Theory",

100: "Redemption, Part I",

101: "Redemption, Part II",

102: "Darmok",

103: "Ensign Ro",

104: "Silicon Avatar",

105: "Disaster",

106: "The Game",

107: "Unification, Part I",

108: "Unification, Part II",

109: "A Matter of Time",

110: "New Ground",

111: "Hero Worship",

112: "Violations",

113: "The Masterpiece Society",

114: "Conundrum",

115: "Power Play",

116: "Ethics",

117: "The Outcast",

118: "Cause and Effect",

119: "The First Duty",

120: "Cost of Living",

121: "The Perfect Mate",

122: "Imaginary Friend",

123: "I, Borg",

124: "The Next Phase",

125: "The Inner Light",

126: "Time's Arrow, Part I",

127: "Time's Arrow, Part II",

128: "Realm of Fear",

129: "Man of the People",

130: "Relics",

131: "Schisms",

132: "True Q",

133: "Rascals",

134: "A Fistful of Datas",

135: "The Quality of Life",

136: "Chain of Command, Part I",

137: "Chain of Command, Part II",

138: "Ship in a Bottle",

139: "Aquiel",

140: "Face of the Enemy",

141: "Tapestry",

142: "Birthright, Part I",

143: "Birthright, Part II",

144: "Starship Mine",

145: "Lessons",

146: "The Chase",

147: "Frame of Mind",

148: "Suspicions",

149: "Rightful Heir",

150: "Second Chances",

151: "Timescape",

152: "Descent, Part I",

153: "Descent, Part II",

154: "Liaisons",

155: "Interface",

156: "Gambit, Part I",

157: "Gambit, Part II",

158: "Phantasms",

159: "Dark Page",

160: "Attached",

161: "Force of Nature",

162: "Inheritance",

163: "Parallels",

164: "The Pegasus",

165: "Homeward",

166: "Sub Rosa",

167: "Lower Decks",

168: "Thine Own Self",

169: "Masks",

170: "Eye of the Beholder",

171: "Genesis",

172: "Journey's End",

173: "Firstborn",

174: "Bloodlines",

175: "Emergence",

176: "Preemptive Strike",

177: "All Good Things... Part I",

178: "All Good Things... Part II"

}
# How many times is this line said?

print('{} times'.format(len(matching_lines)))
for i, (ep, l) in enumerate(matching_lines):

    

    # The +2 handles the 0-index offset and Encounter at Farpoint issue

    ep = 1 if i == 0 else int(ep.split(" ")[1]) + 2

    

    print('[{}]: {}'.format(episodes[ep], l))